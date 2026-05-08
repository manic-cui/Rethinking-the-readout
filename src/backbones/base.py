"""Abstract video-backbone interface for DTP.

Any backbone usable by DTPDetector must subclass DTPBackbone and expose:
  - a single `extract_patch_features(x, layer=None)` returning (B, P, T, D);
  - input spec (img_size, patch_size, num_frames, input_mean, input_std);
  - `max_extract_layer` — highest 0-indexed block that can be tapped.

The stats module is agnostic to P, T, D, so different backbones can have
different patch counts, frame rates (e.g. VideoMAE tubelet-2 halves T), or
embedding widths without any changes to compute_stats / aggregate.
"""
from abc import ABC, abstractmethod
import torch.nn as nn


class DTPBackbone(nn.Module, ABC):
    # Subclasses MUST override these (may also override per-instance in __init__).
    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 8
    num_patches_spatial: int = 196
    embed_dim: int = 768
    input_mean: tuple = (0.45, 0.45, 0.45)
    input_std: tuple = (0.225, 0.225, 0.225)
    max_extract_layer: int = 11

    @abstractmethod
    def extract_patch_features(self, x, layer=None):
        """x: (B, C, T, H, W), channel-normalized as per (input_mean, input_std).

        Returns (B, P, T', D) where T' is the backbone's output temporal dim
        (== self.num_frames for TimeSformer; may be T//2 for VideoMAE tubelet=2).
        If layer is None, returns the final post-norm patch tokens.
        Otherwise taps block `layer` (0-indexed, must be ≤ max_extract_layer).
        """
        ...
