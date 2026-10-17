<div align="center">

# 🎞️ Rethinking the Readout

### Unlocking Video Backbones for AI-Generated Video Detection

**V-PVP: Velocity-Gated Patch Velocity Profiling**

Manni Cui<sup>1,\*</sup> · Ziheng Qin<sup>2,\*</sup> · ZiAn Wang<sup>3</sup> · Ruiqi Liu<sup>2</sup> · Dianyuan Zou<sup>1</sup> · Jianglan Wei<sup>1</sup> · Han Zhou<sup>1</sup> · Yu Liu<sup>1</sup> · Jingrui Xu<sup>1</sup> · Wenhao Wang<sup>4,†</sup> · Zhenyu Zhang<sup>1,†</sup>

<sup>1</sup>Huazhong University of Science and Technology &nbsp; <sup>2</sup>Institute of Automation, Chinese Academy of Sciences &nbsp; <sup>3</sup>Jilin University &nbsp; <sup>4</sup>Vast Intelligence Lab

<sub>\* Equal contribution &nbsp; † Corresponding author</sub>

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://arxiv.org/abs/2607.15321)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.15321)
[![Python](https://img.shields.io/badge/Python-3.13%20tested-yellow.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/manic-cui/Rethinking-the-readout?style=social)](https://github.com/manic-cui/Rethinking-the-readout)

[English](README.md) · [简体中文](README_zh-CN.md)

[📄 Paper](https://arxiv.org/abs/2607.15321) · [🧭 Method](#-how-it-works) · [📊 Results](#-results) · [🚀 Quick Start](#-quick-start) · [📚 Citation](#-citation)

<img src="assets/teaser.jpg" width="95%" alt="The readout is the bottleneck for AIGV detection: patch-level dynamics are averaged away by GAP or CLS readouts.">

</div>

---

> **TL;DR.** Video backbones are not weak at AI-generated video detection; their readout is. Replacing only the GAP/CLS readout with **V-PVP** (~0.5M trainable parameters) lets a **fully frozen** VideoMAE reach **95.28 AUC** on AIGVDBench.

AI-generated videos (AIGVs) typically contain subtle temporal artifacts that arise from inter-frame inconsistencies rather than within individual frames, so video-pretrained backbones should help. In practice, video backbones with standard global readouts often fail to outperform strong image-pretrained probes. We attribute this gap to excessive spatiotemporal aggregation in the readout: compressing each frame into a single global descriptor suppresses local patch-level temporal dynamics and discards inter-patch relations, which are precisely the cues that AIGV detection most reliably depends on.

**V-PVP** keeps the backbone frozen and replaces only the aggregation layer with two parallel streams over the patch velocity field. It works as a plug-and-play readout across video backbones, under both end-to-end fine-tuning and linear probing.

---

## 📰 News

- **2026.09** &nbsp;Accepted to **NeurIPS 2026**.
- **2026.07** &nbsp;Paper released on [arXiv](https://arxiv.org/abs/2607.15321).
- **2026.05** &nbsp;Code and V-PVP head checkpoints released.

---

## ✨ Highlights

| | |
|---|---|
| 🎯 **The readout is the bottleneck** | GAP/CLS readouts erase inter-patch relations and dilute per-channel velocity magnitudes; the backbone representation already carries the signal |
| 🧊 **Frozen backbone** | **95.28** mga AUC on AIGVDBench with VideoMAE fully frozen, above end-to-end fine-tuning with the standard readout (88.34) |
| 🪶 **Lightweight** | 492,674 trainable parameters (~0.5M) |
| 🔌 **Plug-and-play** | Gains across TimeSformer, VideoMAE, UniFormerV2 and Qwen3-VL-8B, and on top of LoRA and full fine-tuning |
| 🏆 **Best on two benchmarks** | 95.28 on AIGVDBench (20 generators) and 93.75 on GenVidBench-143k |

---

## 🧭 How It Works

<div align="center">
<img src="assets/architecture.png" width="95%" alt="Overall architecture of V-PVP">
</div>

A frozen video transformer produces patch tokens $z \in \mathbb{R}^{B \times P \times T' \times D}$. Standard readouts pool these tokens before any change over time is read, which causes two losses:

- **Loss 1: inter-patch relations are erased.** The patch velocity matrix of generated videos has a systematically higher spectral condition number (Cohen's $d \approx 0.64$), and this signal is no longer measurable after aggregation.
- **Loss 2: per-channel velocity magnitudes are diluted.** Real and generated per-channel velocity energies differ by 1.00 vs 0.80 ($d \approx 0.69$) before pooling, but only 1.00 vs 0.88 ($d \approx 0.41$) after.

V-PVP starts from the patch velocity field

$$v_t^{(p)} = z_{t+1}^{(p)} - z_t^{(p)}$$

and reads it through two parallel streams:

1. **Velocity-gated learnable aggregation** (addresses Loss 1). A small MLP scores every patch from its token and velocity magnitude:

$$\alpha_t^{(p)} = \mathrm{softmax}_p\big(\mathrm{MLP}([\,z_t^{(p)},\ \lVert v_t^{(p)} \rVert_2\,])\big), \qquad \tilde z_t = \sum_p \alpha_t^{(p)} z_t^{(p)}$$

2. **Channel-faithful velocity stream** (addresses Loss 2). Taking the absolute value before the spatial mean keeps opposite-signed components from cancelling:

$$r_t = \frac{1}{P}\sum_p \big|v_t^{(p)}\big|$$

Both streams are linearly projected to 128 channels each, concatenated, and passed through GELU → Conv1D (k = 3) → GELU → mean over $t$ → Dropout(0.3) → Linear, giving one logit per clip. The backbone is never updated.

---

## 📊 Results

Mean AUC (%) on the AIGVDBench open-source split (20 generators) and GenVidBench-143k (cross-source protocol). V-PVP uses a frozen VideoMAE-B/16; the VideoMAE row is the same backbone fine-tuned end to end with its standard readout. Numbers are from Tables 1 and 2 of the paper.

| Group | Method | AIGVDBench | GenVidBench-143k |
|---|---|---:|---:|
| AI-generated image detectors | NPR | 89.90 | 50.26 |
| | UnivFD | 90.03 | 78.76 |
| | Effort | 90.29 | 79.08 |
| | DDA | 90.73 | 92.79 |
| | MIRROR | 66.57 | 75.81 |
| AI-generated video detectors | DeMamba | 93.14 | 90.03 |
| | DeCoF | 92.20 | 73.30 |
| | ReSTrav | 69.27 | 90.28 |
| | NSG-VD | 52.55 | 69.22 |
| Video classification models | I3D | 92.85 | 92.76 |
| | SlowFast | 92.70 | 92.95 |
| | TimeSformer | 85.41 | 91.18 |
| | UniFormerV2 | 76.46 | 69.87 |
| | VideoMAE | 88.34 | 88.59 |
| **Ours** | **V-PVP (frozen VideoMAE)** | **95.28** | **93.75** |

**V-PVP composes with backbone adaptation** (mga AUC on AIGVDBench with VideoMAE, Table 4 of the paper):

| Adaptation | Trainable params | Standard readout | V-PVP | Δ |
|---|---:|---:|---:|---:|
| Frozen | ~0.5M | 89.82 | **95.28** | +5.5 |
| LoRA (r = 8) | ~1M | 88.22 | **95.06** | +6.8 |
| Full fine-tune | ~87M | 88.34 | **97.21** | +8.9 |

See the paper for per-generator results, ablations, and results on other backbones.

---

## 📦 Released Checkpoints

| File | Backbone | Protocol | mga AUC |
| ---- | -------- | -------- | ------: |
| `pvp_heads/videomae_k400_vpvp.pt` | VideoMAE-B/16 K400, frozen | AIGVDBench, 20-generator Open-Source test split | **0.9528** |
| `pvp_heads/genvidbench_videomae_auc0.9375.pt` | VideoMAE-B/16 K400, frozen | GenVidBench-143k cross-source protocol | **0.9375** |

Both heads are `state_dict`s of `PVPHead("velocity_gated", D=768)`
(`src/pvp_head.py`, 492,674 parameters) trained with the backbone fully
frozen (no LoRA, no fine-tuning). `eval_opensource.py` implements the
AIGVDBench Open-Source evaluation (3000 real + 60000 fake = 63000 videos);
see [`checkpoints/README.md`](checkpoints/README.md) for the GenVidBench head.

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/manic-cui/Rethinking-the-readout.git
cd Rethinking-the-readout
python -m pip install -r requirements.txt
```

Tested with Python 3.13, `torch` 2.x and `transformers` 4.x / 5.x. The
VideoMAE wrapper loads the checkpoint itself and remaps the attention biases
to the layout of the installed `transformers` version, so both major versions
give identical features.

### 2. Backbone weights

V-PVP uses the public VideoMAE-Base checkpoint fine-tuned on Kinetics-400:
[`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics).
Download the repository as a folder containing `config.json`,
`preprocessor_config.json` and `model.safetensors`, and pass the folder via
`--backbone_weights`.

Patch tokens are taken from the output of the last transformer block, before
any final normalization (the checkpoint uses `use_mean_pooling=True`, so the
encoder has no final LayerNorm). With 16 input frames and `tubelet_size=2`
the backbone yields `T' = 8` temporal tokens per spatial patch, i.e. 7
velocity steps for the head.

### 3. Data

Lay out the AIGVDBench frame folders under a single root:

```
<DATA_ROOT>/
├── Real/test/Real/videos/<vid>/frame_NNN.jpg
└── OpenSource/
    ├── T2V/<gen>/test/videos/<vid>/frame_NNN.jpg
    ├── I2V/<gen>/test/videos/<vid>/frame_NNN.jpg
    └── V2V/<gen>/test/videos/<vid>/frame_NNN.jpg
```

`<gen>` is the generator name (e.g. `Wan2.1`, `Cogvideox1.5`, ...); the 20
generator folders make up the Open-Source split, and categories are
namespaced by modality in the metric reports (e.g. `T2V/Wan2.1`). Each
`<vid>` folder holds the frames of one video as images sorted by filename;
`--num_frames` indices are sampled uniformly over the images present, so use
the same frame extraction for every video.

### 4. Run evaluation

```bash
cd src

python eval_opensource.py \
    --backbone         videomae_k400 \
    --backbone_weights /path/to/videomae-base-finetuned-kinetics \
    --head_ckpt        ../checkpoints/pvp_heads/videomae_k400_vpvp.pt \
    --data_root        /path/to/aigvdbench \
    --num_frames       16 \
    --batch_size       32 \
    --out_json         ../results/videomae_open_source.json
```

This reproduces the V-PVP row of Table 1 (mga AUC 0.9528). The script prints
overall + per-generator AUC and, with `--out_json`, writes a JSON file with
the full per-generator breakdown.

### 5. Load a head in your own code

```python
import torch
from src.pvp_head import PVPHead

head = PVPHead("velocity_gated", D=768)
head.load_state_dict(torch.load("checkpoints/pvp_heads/videomae_k400_vpvp.pt",
                                map_location="cpu"))
head.eval()
```

The head takes `(B, P, T', D)` VideoMAE patch tokens.

---

## 🗂️ Repository Layout

```
.
├── README.md
├── README_zh-CN.md
├── requirements.txt
├── assets/                      figures used in this README
├── src/
│   ├── pvp_head.py              V-PVP head (VelocityGatedAttn + PVPHead)
│   ├── dataset.py               AIGVDBench Open-Source test-split frame loader
│   ├── eval_opensource.py       evaluation entry point
│   └── backbones/
│       ├── __init__.py          BACKBONE_REGISTRY
│       ├── base.py              FrozenVideoBackbone interface
│       └── videomae_k400.py     frozen VideoMAE-B/16 K400 wrapper
└── checkpoints/
    ├── README.md
    └── pvp_heads/
        ├── videomae_k400_vpvp.pt
        └── genvidbench_videomae_auc0.9375.pt
```

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{cui2026rethinking,
  title     = {Rethinking the Readout: Unlocking Video Backbones for {AI}-Generated Video Detection},
  author    = {Cui, Manni and Qin, Ziheng and Wang, ZiAn and Liu, Ruiqi and Zou, Dianyuan and Wei, Jianglan and Zhou, Han and Liu, Yu and Xu, Jingrui and Wang, Wenhao and Zhang, Zhenyu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

---

## 📬 Contact

- **Issues** &nbsp;[github.com/manic-cui/Rethinking-the-readout/issues](https://github.com/manic-cui/Rethinking-the-readout/issues)
- **Email** &nbsp;`mannicui@berkeley.edu`
