"""Frozen video-backbone registry for V-PVP.

Each backbone subclasses :class:`DTPBackbone` and exposes a single
``extract_patch_features(x, layer=None)`` returning ``(B, P, T', D)``.

The four ViT backbones used in the paper are registered below:
  - timesformer_k400   (D=768, P=196, T'=8)
  - videomae_k400      (D=768, P=196, T'=8 with tubelet=2)
  - uniformerv2_k400   (D=768, P=196, T'=8)
  - qwen3vl_8b         (D=1024, P=196, T'=T_in/2)
"""
from .base import DTPBackbone
from .timesformer_k400 import TimeSformerK400
from .videomae_k400 import VideoMAEK400
from .uniformerv2_k400 import UniFormerV2K400
from .qwen3vl_8b import Qwen3VL8BVision

BACKBONE_REGISTRY = {
    'timesformer_k400': TimeSformerK400,
    'videomae_k400':    VideoMAEK400,
    'uniformerv2_k400': UniFormerV2K400,
    'qwen3vl_8b':       Qwen3VL8BVision,
}


def list_backbones():
    return sorted(BACKBONE_REGISTRY.keys())


def build_backbone(name, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list_backbones()}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)


__all__ = ['DTPBackbone', 'BACKBONE_REGISTRY', 'list_backbones', 'build_backbone']
