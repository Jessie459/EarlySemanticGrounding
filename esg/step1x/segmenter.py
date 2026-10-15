import math
import os

import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision.transforms.functional import to_tensor

from ..segment import minmax_normalize
from .conditioner import Qwen25VL_7b_Embedder
from .model import load_models


def input_process_image(img: Image.Image, size_level=512) -> Image.Image:
    """Resize to about size_level x size_level pixels, keeping the aspect ratio (multiples of 16)."""
    w, h = img.size
    r = w / h
    if w > h:
        w_new = math.ceil(math.sqrt(size_level * size_level * r))
        h_new = math.ceil(w_new / r)
    else:
        h_new = math.ceil(math.sqrt(size_level * size_level / r))
        w_new = math.ceil(h_new * r)
    h_new = h_new // 16 * 16
    w_new = w_new // 16 * 16
    return img.resize((w_new, h_new), Image.LANCZOS)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    m = (max_shift - base_shift) / (4096 - 256)
    mu = m * image_seq_len + (base_shift - m * 256)
    return time_shift(mu, 1.0, timesteps).tolist()


class AttentionRecorder:
    """Accumulates the Refined Attention Map (RAM) of the reference-image tokens over all DiT blocks.

    The joint sequence is [text (max_length), noisy latent (hw), reference image (hw)]. For each block, the
    image-to-prompt attention (CAM) is propagated with the row-normalized image-to-image attention (random walk).
    """

    def __init__(self, text_length: int, hw: int):
        self.text_length = text_length
        self.hw = hw
        self.attn = 0
        self.num_blocks = 0

    def add(self, attn_weights: torch.Tensor):
        # attn_weights: [b, heads, seq, seq]
        self_attn = attn_weights[..., -self.hw :, -self.hw :]
        cross_attn = attn_weights[..., -self.hw :, : self.text_length]
        self_attn = self_attn / (self_attn.sum(dim=-1, keepdim=True) + 1e-6)
        ref_cross_attn = self_attn @ cross_attn
        self.attn = self.attn + ref_cross_attn[0].mean(0)  # [hw, text_length]
        self.num_blocks += 1

    def get(self) -> torch.Tensor:
        return (self.attn / self.num_blocks).float()


class Step1XSegmenter:
    """Zero-shot RIS with Step1X-Edit v1.1 from a single denoising step (t = 0)."""

    def __init__(
        self,
        model_root,
        qwen_path,
        device="cuda",
        feat_block=40,
        size_level=512,
        num_steps=28,
        seed=42,
        max_length=640,
        load_text_encoder=True,
        load_dit=True,
    ):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.feat_block = feat_block
        self.size_level = size_level
        self.num_steps = num_steps
        self.seed = seed
        self.max_length = max_length

        # Text encoder and DiT can be loaded separately to fit GPUs with < 48 GB (see evaluate.py --low_memory).
        if load_dit:
            self.ae, self.dit = load_models(
                dit_path=os.path.join(model_root, "step1x-edit-v1p1-official.safetensors"),
                ae_path=os.path.join(model_root, "vae.safetensors"),
            )
            self.ae, self.dit = self.ae.to(self.device), self.dit.to(self.device)
        if load_text_encoder:
            self.encoder = Qwen25VL_7b_Embedder(qwen_path, max_length=max_length, dtype=self.dtype, device=self.device)

    def prepare(self, x, ref_latent, txt, mask):
        _, _, h, w = x.shape
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=1)

        ref_img_ids = torch.ones(h // 2, w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "h w c -> b (h w) c", b=1)

        return {
            "img": torch.cat([img, ref_img.to(img)], dim=-2),
            "img_ids": torch.cat([img_ids, ref_img_ids], dim=-2).to(self.device),
            "llm_embedding": txt.to(self.device),
            "mask": mask.to(self.device),
            "txt_ids": torch.zeros(1, txt.shape[1], 3, device=self.device),
        }

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = input_process_image(image.convert("RGB"), size_level=self.size_level)
        return to_tensor(image)[None].to(self.device)

    @torch.inference_mode()
    def encode_text(self, image: Image.Image, instruction: str) -> dict:
        """Qwen2.5-VL embedding of the instruction (only the valid tokens) and the range of instruction tokens."""
        txt, mask, tokens = self.encoder([instruction], self.preprocess(image))
        tokens = tokens[0]
        n = int(mask.sum())
        return {
            "txt": txt[:, :n],
            "token_range": (tokens.index("<|vision_end|>") + 1, tokens.index("<|im_end|>")),
        }

    @torch.inference_mode()
    def segment(self, image: Image.Image, text: dict):
        """Returns the attention prior [h, w] in [0, 1] and the image features [C, h, w] on the 16x16 token grid."""
        image = self.preprocess(image)
        _, _, height, width = image.shape
        h, w = height // 16, width // 16

        n = text["txt"].shape[1]
        txt = torch.zeros(1, self.max_length, text["txt"].shape[2], dtype=self.dtype, device=self.device)
        txt[:, :n] = text["txt"].to(self.device)
        mask = torch.zeros(1, self.max_length, dtype=torch.long, device=self.device)
        mask[:, :n] = 1

        # The VAE samples its posterior; seed it so that results are reproducible.
        torch.manual_seed(self.seed)
        with torch.autocast("cuda", dtype=self.dtype):
            ref_latent = self.ae.encode(image * 2 - 1)

        rng = torch.Generator(device=self.device).manual_seed(self.seed)
        x = torch.randn((1, 16, height // 8, width // 8), device=self.device, dtype=self.dtype, generator=rng)
        t = get_schedule(self.num_steps, h * w)[0]  # first denoising step (t = 0 in the paper's notation)
        inputs = self.prepare(x, ref_latent, txt, mask)

        recorder = AttentionRecorder(self.max_length, h * w)
        with torch.autocast("cuda", dtype=self.dtype):
            t_vec = torch.full((1,), t, dtype=inputs["img"].dtype, device=self.device)
            _, feat = self.dit(**inputs, timesteps=t_vec, t_vec=t_vec, recorder=recorder, feat_block=self.feat_block)

        # Attention prior: normalize the map of every instruction token, then average over tokens.
        attn = recorder.get().reshape(h, w, self.max_length)
        attn = torch.stack([minmax_normalize(attn[:, :, i]) for i in range(*text["token_range"])]).mean(0)
        prior = minmax_normalize(attn)

        feat = rearrange(feat[0, h * w :], "(h w) c -> c h w", h=h, w=w)  # reference-image tokens
        return prior, feat

    def __call__(self, image: Image.Image, instruction: str):
        return self.segment(image, self.encode_text(image, instruction))
