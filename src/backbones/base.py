"""Abstract frozen video-backbone interface used by the V-PVP head.

A backbone must expose:
  - ``extract_patch_features(x, layer=None)`` returning patch tokens of shape
    ``(B, P, T', D)``;
  - its input spec (``img_size``, ``patch_size``, ``num_frames``,
    ``input_mean``, ``input_std``);
  - ``max_extract_layer``, the highest 0-indexed block that can be tapped.

The V-PVP head is agnostic to P, T' and D.
"""
from abc import ABC, abstractmethod

import torch.nn as nn


class FrozenVideoBackbone(nn.Module, ABC):
    # Subclasses override these (usually per instance in __init__).
    img_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    num_patches_spatial: int = 196
    embed_dim: int = 768
    input_mean: tuple = (0.485, 0.456, 0.406)
    input_std: tuple = (0.229, 0.224, 0.225)
    max_extract_layer: int = 11

    @abstractmethod
    def extract_patch_features(self, x, layer=None):
        """x: (B, C, T, H, W), channel-normalized with (input_mean, input_std).

        Returns (B, P, T', D), where T' is the backbone's output temporal
        length (T // 2 for VideoMAE with tubelet_size=2). With ``layer=None``
        the tokens are the output of the last transformer block, taken before
        any final normalization. Otherwise block ``layer`` (0-indexed, must be
        <= ``max_extract_layer``) is tapped.
        """
        ...
