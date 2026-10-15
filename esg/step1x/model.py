# Modified from Step1X-Edit (https://github.com/stepfun-ai/Step1X-Edit), licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
import math
from dataclasses import dataclass

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from .autoencoder import AutoEncoder
from .connector import Qwen2Connector
from .layers import DoubleStreamBlock, EmbedND, LastLayer, MLPEmbedder, SingleStreamBlock


@dataclass
class Step1XParams:
    in_channels: int = 64
    out_channels: int = 64
    vec_in_dim: int = 768
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: tuple[int, ...] = (16, 56, 56)
    theta: int = 10_000
    qkv_bias: bool = True
    version: str = "v1.1"


class Step1XEdit(nn.Module):
    """Step1X-Edit DiT. Blocks are indexed 0-18 (double stream) and 19-56 (single stream)."""

    def __init__(self, params: Step1XParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        pe_dim = params.hidden_size // params.num_heads
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=list(params.axes_dim))
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias
                )
                for _ in range(params.depth)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.connector = Qwen2Connector(version=params.version)

    @staticmethod
    def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def forward(self, img, img_ids, txt_ids, timesteps, llm_embedding, t_vec, mask, recorder=None, feat_block=None):
        """Returns the velocity prediction and the MMSA output of block `feat_block` (all tokens after the text)."""
        txt, y = self.connector(llm_embedding, t_vec, mask)

        img = self.img_in(img)
        vec = self.time_in(self.timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        feat = None
        for block_idx, block in enumerate(self.double_blocks):
            img, txt, f = block(
                img=img, txt=txt, vec=vec, pe=pe, recorder=recorder, return_feat=block_idx == feat_block
            )
            feat = f if f is not None else feat

        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)
        for block_idx, block in enumerate(self.single_blocks, start=len(self.double_blocks)):
            img, f = block(img, vec=vec, pe=pe, recorder=recorder, return_feat=block_idx == feat_block)
            feat = f[:, txt_len:] if f is not None else feat
        img = img[:, txt_len:, ...]

        img = self.final_layer(img, vec)
        return img, feat


def load_state_dict(model, ckpt_path):
    state_dict = load_file(ckpt_path, "cpu")
    model.load_state_dict(state_dict, strict=False, assign=True)
    return model


def load_models(dit_path, ae_path):
    with torch.device("meta"):
        ae = AutoEncoder(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )
    ae = load_state_dict(ae, ae_path).to(dtype=torch.float32)

    with torch.device("meta"):
        dit = Step1XEdit(Step1XParams())
    dit = load_state_dict(dit, dit_path).to(dtype=torch.bfloat16)
    return ae, dit
