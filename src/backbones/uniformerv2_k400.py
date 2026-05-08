"""UniFormerV2-B/16 K400 backbone wrapper.

UniFormerV2 augments a CLIP ViT-B/16 with (optionally) Local MHRA around each
attention block and a separate "global block" bank that fuses information into
a single temporal cls token. For the K400 + K710 8x224 checkpoint:
  - N_LAYERS: 4 global blocks (unused for DTP — we only want patch tokens)
  - NO_LMHRA: True (no 3D-conv wrapper around attention)
  - TEMPORAL_DOWNSAMPLE: False (T preserved in the conv1 stem)

So effectively the backbone is a plain CLIP ViT-B/16 with:
  - 3D conv stem (kernel (1, 16, 16), stride (1, 16, 16)) → patches per frame
  - 12 transformer blocks operating on tokens (L=197, N*T, C=768)
  - We bypass the global-cls head entirely and read patch tokens directly.

Output: (B, P=196, T=8, D=768).

Checkpoint source: `Andy1621/uniformerv2` on HuggingFace (file
`k400+k710_uniformerv2_b16_8x224.pyth`).
"""
import torch

from .base import DTPBackbone
from ._vendored.uniformerv2 import uniformerv2_b16


_NATIVE_T = 8


class UniFormerV2K400(DTPBackbone):
    """Frozen UniFormerV2-B/16, 8-frame 224 K400+K710. 12 transformer blocks."""
    max_extract_layer = 11  # 12 resblocks, 0-indexed

    def __init__(self, pretrained_model, num_frames=8, img_size=224, patch_size=16):
        super().__init__()
        if img_size != 224:
            raise ValueError(f"UniFormerV2-B/16 8x224 trained at 224; got {img_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_spatial = (img_size // patch_size) ** 2  # 196
        self.embed_dim = 768
        # UniFormerV2 is initialized from OpenAI CLIP ViT-B/16 → CLIP normalization.
        # OpenAI CLIP uses (0.48145466, 0.4578275, 0.40821073) / (0.26862954, ...)
        self.input_mean = (0.48145466, 0.4578275, 0.40821073)
        self.input_std  = (0.26862954, 0.26130258, 0.27577711)

        self.net = uniformerv2_b16(
            pretrained=False,                 # don't need CLIP-ViT init — K400 ckpt has them
            t_size=_NATIVE_T,
            no_lmhra=True,
            temporal_downsample=False,
            return_list=[8, 9, 10, 11],
            n_layers=4,
            num_classes=400,
            frozen=False,
        )
        ckpt = torch.load(pretrained_model, map_location='cpu', weights_only=False)
        # Training code wraps the backbone under `self.backbone = ...` so state keys
        # start with `backbone.`. Strip that for our bare VisionTransformer.
        state_dict = {
            (k[len('backbone.'):] if k.startswith('backbone.') else k): v
            for k, v in ckpt.items()
        }
        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"UniFormerV2 load missing keys: {missing[:5]} ...")

        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()
        return self

    @staticmethod
    def _temporal_resample(x, target_T):
        T = x.shape[2]
        if T == target_T:
            return x
        idx = torch.linspace(0, T - 1, target_T, device=x.device).round().long()
        return x.index_select(2, idx)

    @torch.no_grad()
    def extract_patch_features(self, x, layer=None):
        """x: (B, C, T, H, W) → (B, P=196, T=8, D=768)."""
        if layer is None:
            layer = self.max_extract_layer     # final (block 11)
        if layer < 0 or layer > self.max_extract_layer:
            raise ValueError(
                f"layer {layer} out of range [0, {self.max_extract_layer}]"
            )

        x8 = self._temporal_resample(x, _NATIVE_T)
        N, C, T, H, W = x8.shape

        # --- Mirror VisionTransformer.forward up through the resblocks ---
        h = self.net.conv1(x8)                  # (N, 768, T, H', W')
        _, C2, _, Hp, Wp = h.shape
        h = h.permute(0, 2, 3, 4, 1).reshape(N * T, Hp * Wp, C2)
        # Add CLS + pos_embed
        cls_tok = self.net.class_embedding.to(h.dtype) \
            + torch.zeros(h.shape[0], 1, h.shape[-1], dtype=h.dtype, device=h.device)
        h = torch.cat([cls_tok, h], dim=1)      # (N*T, L=197, C)
        h = h + self.net.positional_embedding.to(h.dtype)
        h = self.net.ln_pre(h)
        h = h.permute(1, 0, 2)                  # (L, N*T, C)

        # Run the requested number of resblocks (no global blocks, no cls fuse).
        for i in range(layer + 1):
            h = self.net.transformer.resblocks[i](h, T)

        # Drop CLS, keep patch tokens → (L-1, N*T, C) → (N, T, P, C) → (N, P, T, C)
        patch = h[1:]                           # (P=196, N*T, C)
        P = patch.shape[0]
        patch = patch.view(P, N, T, C2)         # (P, N, T, C)
        patch = patch.permute(1, 0, 2, 3).contiguous()   # (N, P, T, C)
        return patch
