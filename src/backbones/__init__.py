"""Frozen video-backbone registry for V-PVP.

Each backbone exposes ``extract_patch_features(x, layer=None)`` returning
patch tokens of shape ``(B, P, T', D)``.

This release ships the VideoMAE backbone used for the main results:
  - videomae_k400   VideoMAE-B/16 fine-tuned on Kinetics-400
                    (D=768, P=196, T'=8 for 16 input frames, tubelet_size=2)
"""
from .base import FrozenVideoBackbone
from .videomae_k400 import VideoMAEK400

BACKBONE_REGISTRY = {
    'videomae_k400': VideoMAEK400,
}


def list_backbones():
    return sorted(BACKBONE_REGISTRY.keys())


def build_backbone(name, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list_backbones()}"
        )
    return BACKBONE_REGISTRY[name](**kwargs)


__all__ = ['FrozenVideoBackbone', 'BACKBONE_REGISTRY', 'list_backbones', 'build_backbone']
