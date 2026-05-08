"""Qwen3-VL-8B vision-tower backbone wrapper.

Loads ONLY the vision encoder (`model.visual.*` keys) of Qwen3-VL-8B; the
LLM weights are skipped. Returns post-block / pre-merger patch tokens at
hidden_size=1024.

Vision spec (from Qwen3-VL config.json → vision_config):
  - depth=24, hidden_size=1024, num_heads=16, intermediate_size=4096
  - patch_size=16, temporal_patch_size=2, spatial_merge_size=2
  - in_channels=3, num_position_embeddings=2304
  - hidden_act='gelu_pytorch_tanh'
  - input_mean=input_std=(0.5,)*3 (matches vision-tower preprocessing)

Output shape: (B, P=196, T'=T_in/2, D=1024) for 224×224 input.
Tubelet-2 halves the temporal axis (matches VideoMAE convention).

Input ordering follows Qwen2/3-VL's merge-aware patch flattening:
  for t in T_grid:
    for h_outer in H_grid//ms:
      for w_outer in W_grid//ms:
        for h_inner, w_inner in (ms × ms):
          yield patch(t, h_outer*ms+h_inner, w_outer*ms+w_inner)
"""
import json
import os

import torch
import torch.nn as nn

from .base import DTPBackbone


_TUBELET = 2
_MERGE = 2


class Qwen3VL8BVision(DTPBackbone):
    """Frozen Qwen3-VL-8B vision tower (24 blocks, D=1024)."""

    max_extract_layer = 23  # 24 blocks, 0-indexed

    def __init__(self, pretrained_model, num_frames=8, img_size=224, patch_size=16):
        super().__init__()
        if num_frames % _TUBELET != 0:
            raise ValueError(f"num_frames must be divisible by {_TUBELET}, got {num_frames}")
        if (img_size // patch_size) % _MERGE != 0:
            raise ValueError(
                f"img_size/patch_size = {img_size//patch_size} must be divisible by "
                f"merge_size={_MERGE}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_spatial = (img_size // patch_size) ** 2
        self.embed_dim = 1024
        self.input_mean = (0.5, 0.5, 0.5)
        self.input_std = (0.5, 0.5, 0.5)

        # Lazy-import HF so non-Qwen runs don't pay the import cost.
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        full_cfg = Qwen3VLConfig.from_pretrained(pretrained_model)
        self.vision = Qwen3VLVisionModel(full_cfg.vision_config)
        self._load_visual_weights(pretrained_model)

        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

    def _load_visual_weights(self, pretrained_model):
        """Load only `model.visual.*` keys from the (possibly sharded) checkpoint."""
        from safetensors.torch import load_file as safe_load

        idx_path = os.path.join(pretrained_model, 'model.safetensors.index.json')
        sd = {}
        if os.path.isfile(idx_path):
            wm = json.load(open(idx_path))['weight_map']
            visual_keys = {k: v for k, v in wm.items() if k.startswith('model.visual.')}
            for shard in sorted(set(visual_keys.values())):
                shard_sd = safe_load(os.path.join(pretrained_model, shard))
                for k, v in shard_sd.items():
                    if k in visual_keys:
                        sd[k.replace('model.visual.', '')] = v
        else:
            single = os.path.join(pretrained_model, 'model.safetensors')
            for k, v in safe_load(single).items():
                if k.startswith('model.visual.'):
                    sd[k.replace('model.visual.', '')] = v

        if not sd:
            raise RuntimeError(f'No visual weights found under {pretrained_model}')

        missing, unexpected = self.vision.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f'Unexpected keys loading Qwen3VL vision: {unexpected[:5]}')
        # `rotary_pos_emb.inv_freq` etc. are runtime buffers; not in checkpoint.
        real_missing = [k for k in missing if 'inv_freq' not in k]
        if real_missing:
            raise RuntimeError(f'Missing param keys loading Qwen3VL vision: {real_missing[:5]}')

    def train(self, mode=True):
        super().train(mode)
        self.vision.eval()
        return self

    def _patchify(self, x):
        """(B, C, T, H, W) → (B*N, patch_dim) in Qwen merge-aware order.

        Returns (flat, T_grid, H_grid, W_grid).
        """
        B, C, T, H, W = x.shape
        Pp = self.patch_size
        T_grid, H_grid, W_grid = T // _TUBELET, H // Pp, W // Pp
        x = x.view(B, C, T_grid, _TUBELET, H_grid, Pp, W_grid, Pp)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x = x.view(B, T_grid, H_grid // _MERGE, _MERGE, W_grid // _MERGE, _MERGE,
                   C * _TUBELET * Pp * Pp)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        x = x.view(-1, C * _TUBELET * Pp * Pp)
        return x, T_grid, H_grid, W_grid

    def _unpatch(self, last, B, T_grid, H_grid, W_grid):
        """(B*T_grid*H_grid*W_grid, D) → (B, P=H_grid*W_grid, T_grid, D)."""
        D = last.size(-1)
        x = last.view(B, T_grid, H_grid // _MERGE, W_grid // _MERGE,
                      _MERGE, _MERGE, D)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()  # undo
        x = x.view(B, T_grid, H_grid * W_grid, D)
        return x.permute(0, 2, 1, 3).contiguous()  # (B, P, T_grid, D)

    @torch.no_grad()
    def extract_patch_features(self, x, layer=None):
        if layer is not None:
            raise NotImplementedError(
                "Per-layer tapping is not implemented for Qwen3VL backbone yet.")
        B = x.size(0)
        flat, T_grid, H_grid, W_grid = self._patchify(x)
        # Match the vision tower's parameter dtype (loaded from bf16 checkpoint).
        param_dtype = next(self.vision.parameters()).dtype
        flat = flat.to(param_dtype)
        grid_thw = torch.tensor(
            [[T_grid, H_grid, W_grid]] * B, dtype=torch.long, device=x.device)
        out = self.vision(flat, grid_thw)
        last = out.last_hidden_state.to(torch.float32)
        return self._unpatch(last, B, T_grid, H_grid, W_grid)
