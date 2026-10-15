import inspect
import json
import math
import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen3Model

from ..segment import minmax_normalize
from .transformer import Flux2DiT
from .vae import Flux2VAE

HIDDEN_STATES_LAYERS = (9, 18, 27)  # Qwen3 layers used as the FLUX.2-klein text embedding


def load_safetensors_dir(path):
    state_dict = {}
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            state_dict.update(load_file(os.path.join(path, name)))
    return state_dict


class Flux2Recorder:
    """Accumulates the Refined Attention Map (RAM) of the reference-image tokens over all DiT blocks and stores the
    MMSA output of one block.

    The joint sequence is [text, noisy latent (hw), reference image (hw)].
    """

    def __init__(self, img_len: int, feat_block: int):
        self.img_len = img_len
        self.feat_block = feat_block
        self.block = -1
        self.attn = None
        self.feat = None

    def add_attn(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: [b, heads, seq, d] (after RoPE)
        q = q[:, :, -self.img_len :]
        attn = (torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))).softmax(dim=-1)
        txt_len = attn.shape[-1] - 2 * self.img_len
        self_attn = attn[..., -self.img_len :]
        cross_attn = attn[..., :txt_len]
        self_attn = self_attn / (self_attn.sum(dim=-1, keepdim=True) + 1e-6)
        ref_cross_attn = (self_attn @ cross_attn)[0].mean(0)  # [hw, txt_len]
        self.attn = ref_cross_attn if self.attn is None else self.attn + ref_cross_attn

    def add_feat(self, attn_output: torch.Tensor):
        if self.block == self.feat_block:
            self.feat = attn_output[0, -self.img_len :].clone()


class Flux2Segmenter:
    """Zero-shot RIS with FLUX.2-klein from a single denoising step (t = 0)."""

    def __init__(self, model_root, device="cuda", feat_block=20, resolution=1024, seed=42, max_sequence_length=512):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.feat_block = feat_block
        self.resolution = resolution
        self.seed = seed
        self.max_sequence_length = max_sequence_length

        with open(os.path.join(model_root, "transformer", "config.json")) as f:
            cfg = json.load(f)
        cfg = {k: v for k, v in cfg.items() if k in inspect.signature(Flux2DiT).parameters}
        with torch.device("meta"):
            dit = Flux2DiT(**cfg)
        dit.load_state_dict(load_safetensors_dir(os.path.join(model_root, "transformer")), assign=True)
        self.dit = dit.to(device=self.device, dtype=self.dtype).eval()

        with torch.device("meta"):
            vae = Flux2VAE()
        vae.load_state_dict(load_safetensors_dir(os.path.join(model_root, "vae")), assign=True)
        self.vae = vae.to(device=self.device, dtype=self.dtype).eval()

        self.text_encoder = Qwen3Model.from_pretrained(
            os.path.join(model_root, "text_encoder"), torch_dtype=self.dtype
        )
        self.text_encoder = self.text_encoder.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_root, "tokenizer"))

    def encode_prompt(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_sequence_length
        )
        input_ids = inputs["input_ids"].to(self.device)
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(self.device),
            output_hidden_states=True,
            use_cache=False,
        )
        out = torch.stack([output.hidden_states[k] for k in HIDDEN_STATES_LAYERS], dim=1).to(self.dtype)
        prompt_embeds = rearrange(out, "b c l d -> b l (c d)")
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        return prompt_embeds, tokens

    @staticmethod
    def position_ids(t: int, h: int, w: int, l: int) -> torch.Tensor:
        """(t, h, w, l) coordinates for RoPE, shape [1, t*h*w*l, 4]."""
        return torch.cartesian_prod(torch.arange(t), torch.arange(h), torch.arange(w), torch.arange(l))[None]

    @torch.inference_mode()
    def __call__(self, image: Image.Image, instruction: str):
        """Returns the attention prior [h, w] in [0, 1] and the image features [C, h, w] on the 16x16 token grid."""
        size = self.resolution
        image = image.convert("RGB").resize((size, size), resample=Image.Resampling.LANCZOS)
        h = w = size // 16

        prompt_embeds, tokens = self.encode_prompt(instruction)
        # Instruction tokens: skip '<|im_start|>' 'user' '\n' and the trailing period.
        i, j = 3, tokens.index("<|im_end|>")
        if tokens[j - 1].endswith("."):
            j -= 1

        pixels = torch.tensor(np.array(image, dtype=np.float32), dtype=self.dtype, device=self.device)
        pixels = rearrange(pixels * (2 / 255) - 1, "h w c -> 1 c h w")
        ref_latents = rearrange(self.vae.encode(pixels), "b c h w -> b (h w) c")

        rng = torch.Generator("cuda").manual_seed(self.seed)
        noise = torch.randn((1, 128, h, w), generator=rng, device="cuda", dtype=self.dtype).to(self.device)
        noise = rearrange(noise, "b c h w -> b (h w) c")

        text_ids = self.position_ids(1, 1, 1, prompt_embeds.shape[1])
        img_ids = torch.cat(
            [self.position_ids(1, h, w, 1), self.position_ids(1, h, w, 1) + torch.tensor([10, 0, 0, 0])], dim=1
        )

        recorder = Flux2Recorder(img_len=h * w, feat_block=self.feat_block)
        self.dit(
            hidden_states=torch.cat([noise, ref_latents], dim=1),
            timestep=torch.ones(1, dtype=self.dtype, device=self.device),  # sigma = 1 at the first denoising step
            guidance=torch.tensor([4.0], device=self.device),
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids.to(self.device),
            img_ids=img_ids.to(self.device),
            recorder=recorder,
        )

        attn = recorder.attn[:, i:j].float().mean(dim=-1).reshape(h, w)
        prior = minmax_normalize(attn)
        feat = rearrange(recorder.feat, "(h w) c -> c h w", h=h, w=w)
        return prior, feat
