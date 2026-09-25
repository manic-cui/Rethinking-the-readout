"""AIGVDBench Open-Source split dataset (test only, frame-folder layout).

Expected directory structure under ``data_root``::

    Real/test/Real/videos/<vid>/frame_NNN.{jpg,png}                       (real)
    OpenSource/<modality>/<gen>/test/videos/<vid>/frame_NNN.{jpg,png}     (fake)
    OpenSource/<modality>/<gen>/videos/<vid>/...        (fallback for some gens)

where ``<modality>`` is one of ``T2V``, ``I2V``, ``V2V`` and ``<gen>`` is the
generator name (e.g. ``Wan2.1``, ``CogVideoX``, ...). Each video is a folder
of frame images (one image per frame).

Frames are uniformly sub-sampled to ``num_frames`` indices, center-cropped to
the shorter side, resized to ``crop_size``, and normalized with the dataset
mean/std (0.45 / 0.225; the eval loop later renormalizes to the backbone's
expected statistics).
"""
import glob
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


MEAN = (0.45, 0.45, 0.45)
STD  = (0.225, 0.225, 0.225)
IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")
TEST_MODALITIES = ("T2V", "I2V", "V2V")


@dataclass
class OpenSourceConfig:
    data_root: str
    num_frames: int = 8
    crop_size: int = 224
    generator_filter: List[str] = None
    max_per_category: int = 0   # 0 = no cap


def _scan(data_root: str, generator_filter, max_per_cat: int):
    """Return list of (video_dir, label, category_name) tuples for the test
    set. Categories are namespaced by modality (e.g. ``T2V/Wan2.1``)."""
    if not os.path.isdir(data_root):
        raise RuntimeError(f"data_root not found: {data_root}")

    out = []

    real_dir = os.path.join(data_root, "Real", "test", "Real", "videos")
    if os.path.isdir(real_dir):
        dirs = sorted(os.listdir(real_dir))
        if max_per_cat > 0:
            dirs = dirs[:max_per_cat]
        for v in dirs:
            p = os.path.join(real_dir, v)
            if os.path.isdir(p):
                out.append((p, 0, "real"))

    for modality in TEST_MODALITIES:
        mod_dir = os.path.join(data_root, "OpenSource", modality)
        if not os.path.isdir(mod_dir):
            continue
        for gen in sorted(os.listdir(mod_dir)):
            cat_name = f"{modality}/{gen}"
            if generator_filter and (gen not in generator_filter
                                     and cat_name not in generator_filter):
                continue
            gen_dir = None
            for candidate in (
                os.path.join(mod_dir, gen, "test", "videos"),
                os.path.join(mod_dir, gen, "videos"),
            ):
                if os.path.isdir(candidate):
                    gen_dir = candidate
                    break
            if gen_dir is None:
                continue
            dirs = sorted(os.listdir(gen_dir))
            if max_per_cat > 0:
                dirs = dirs[:max_per_cat]
            for v in dirs:
                p = os.path.join(gen_dir, v)
                if os.path.isdir(p):
                    out.append((p, 1, cat_name))
    return out


class OpenSourceTest(torch.utils.data.Dataset):
    """Test-split frame loader. Returns ``(clip, label, idx, category)``."""
    def __init__(self, cfg: OpenSourceConfig):
        self.cfg = cfg
        self.samples = _scan(cfg.data_root,
                             cfg.generator_filter or [],
                             cfg.max_per_category)
        if not self.samples:
            raise RuntimeError(f"no test samples found under {cfg.data_root}")

    def __len__(self):
        return len(self.samples)

    def _frames(self, video_dir):
        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(video_dir, ext)))
        return sorted(paths)

    def _indices(self, total):
        T = self.cfg.num_frames
        if total < T:
            return list(range(total)) + [total - 1] * (T - total)
        return np.linspace(0, total - 1, T, dtype=int).tolist()

    def __getitem__(self, idx):
        video_dir, label, category = self.samples[idx]
        paths = self._frames(video_dir)
        if not paths:
            raise RuntimeError(f"No frames in {video_dir}")
        indices = self._indices(len(paths))
        imgs = [Image.open(paths[i]).convert("RGB") for i in indices]

        W, H = imgs[0].size
        s  = min(W, H)
        cs = self.cfg.crop_size
        cx = (W - s) // 2
        cy = (H - s) // 2

        tensors = []
        for im in imgs:
            im = TF.crop(im, cy, cx, s, s)
            im = TF.resize(im, (cs, cs), interpolation=TF.InterpolationMode.BICUBIC)
            t  = TF.to_tensor(im)
            t  = TF.normalize(t, MEAN, STD)
            tensors.append(t)
        clip = torch.stack(tensors, dim=1)   # (C, T, H, W)
        return clip, label, idx, category


def collate(batch):
    clips  = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    cats   = [b[3] for b in batch]
    return clips, labels, cats
