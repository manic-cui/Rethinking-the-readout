"""TimeSformer (divided_space_time) K400 backbone wrapper."""
from functools import partial

import torch
import torch.nn as nn

from .base import DTPBackbone
from .vit import VisionTransformer, default_cfgs, _conv_filter
from .weights import load_pretrained


class TimeSformerK400(DTPBackbone):
    """Frozen Kinetics-400-pretrained TimeSformer, 12 blocks, D=768.

    Checkpoint: TimeSformer_divST_8x32_224_K400.pyth.
    """
    max_extract_layer = 11  # 12 blocks, 0-indexed

    def __init__(self, pretrained_model, num_frames=8, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_spatial = (img_size // patch_size) ** 2
        self.embed_dim = 768
        # TimeSformer K400 mean/std (from the reference training config):
        self.input_mean = (0.45, 0.45, 0.45)
        self.input_std  = (0.225, 0.225, 0.225)

        self.vit = VisionTransformer(
            img_size=img_size, num_classes=400, patch_size=patch_size,
            embed_dim=self.embed_dim, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
            num_frames=num_frames, attention_type='divided_space_time',
        )
        self.vit.default_cfg = default_cfgs['vit_base_patch16_224']
        load_pretrained(
            self.vit, num_classes=400, in_chans=3, filter_fn=_conv_filter,
            img_size=img_size, num_frames=num_frames,
            num_patches=self.num_patches_spatial,
            attention_type='divided_space_time',
            pretrained_model=pretrained_model,
        )
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def extract_patch_features(self, x, layer=None):
        return self.vit.forward_features(
            x, return_patch_features=True, extract_layer=layer,
        )
