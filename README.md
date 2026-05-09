# V-PVP: Velocity-based Patch Velocity Profiling

This is the anonymous code release for **V-PVP**, a lightweight, frozen-backbone
head for AI-generated video detection. Given any pre-trained video transformer
that returns patch tokens `z ∈ ℝ^{B×P×T'×D}`, V-PVP computes the velocity
field `v[t] = z[t+1] - z[t]`, applies a velocity-gated spatial attention to
get a context vector `c̃[t]`, complements it with a per-channel velocity
magnitude stream `r[t] = (1/P) Σ_p |v[t,p]|`, and feeds the dual-stream tensor
through a small temporal Conv1D + linear classifier (~0.5 M trainable
parameters).

The release contains:
- `src/` — model implementation (head + four ViT backbones + dataset + eval)
- `checkpoints/pvp_heads/` — released V-PVP head weights for the four ViT
  backbones reported in the paper

## Released checkpoints

| Backbone           | head ckpt                                  | mga AUC (Open-Source) |
| ------------------ | ------------------------------------------ | --------------------: |
| TimeSformer-Base   | `pvp_heads/timesformer_k400_vpvp.pt`       | **0.9124**            |
| UniFormerV2-Base   | `pvp_heads/uniformerv2_k400_vpvp.pt`       | **0.9356**            |
| VideoMAE-Base      | `pvp_heads/videomae_k400_vpvp.pt`          | **0.9528**            |
| Qwen3-VL-8B (vision tower) | `pvp_heads/qwen3vl_8b_vpvp.pt`     | **0.9778**            |

All heads are trained with a **frozen** backbone (no LoRA, no fine-tuning),
and evaluated on the 20-generator Open-Source test split (3000 real +
60000 fake = 63000 videos).

## Install

```bash
python -m pip install -r requirements.txt
```

Tested with `python>=3.9`, `torch>=2.1`, `transformers>=4.46` (needed for
Qwen3-VL).

## Backbone weights

V-PVP uses **public** Kinetics-400 / vision-tower checkpoints. Download them
from the original sources and pass the path via `--backbone_weights`:

| Backbone           | Source                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------- |
| TimeSformer-Base   | TimeSformer official release: `TimeSformer_divST_8x32_224_K400.pyth`                    |
| UniFormerV2-Base   | UniFormerV2 release on HuggingFace: `k400+k710_uniformerv2_b16_8x224.pyth`              |
| VideoMAE-Base      | HuggingFace: `MCG-NJU/videomae-base-finetuned-kinetics` (download the repo as a folder) |
| Qwen3-VL-8B        | HuggingFace: `Qwen/Qwen3-VL-8B-Instruct` (download the repo; only `model.visual.*` weights are loaded) |

## Data

Lay out AIGVDBench under a single root as follows:

```
<DATA_ROOT>/
├── Real/test/Real/videos/<vid>/frame_NNN.jpg
└── OpenSource/
    ├── T2V/<gen>/test/videos/<vid>/frame_NNN.jpg
    ├── I2V/<gen>/test/videos/<vid>/frame_NNN.jpg
    └── V2V/<gen>/test/videos/<vid>/frame_NNN.jpg
```

`<gen>` is the generator name (e.g. `Wan2.1`, `CogVideoX`, ...). 20 generator
folders make up the Open-Source split; categories are namespaced by modality
in metric reports (e.g. `T2V/Wan2.1`).

## Run evaluation

```bash
cd src

# VideoMAE (the 0.9528 row in Tab. 1)
python eval_opensource.py \
    --backbone         videomae_k400 \
    --backbone_weights /path/to/videomae-base-finetuned-kinetics \
    --head_ckpt        ../checkpoints/pvp_heads/videomae_k400_vpvp.pt \
    --data_root        /path/to/aigvdbench \
    --num_frames       16 \
    --batch_size       32 \
    --out_json         ../results/videomae_open_source.json

# TimeSformer
python eval_opensource.py \
    --backbone         timesformer_k400 \
    --backbone_weights /path/to/TimeSformer_divST_8x32_224_K400.pyth \
    --head_ckpt        ../checkpoints/pvp_heads/timesformer_k400_vpvp.pt \
    --data_root        /path/to/aigvdbench \
    --num_frames       8 \
    --batch_size       32

# UniFormerV2
python eval_opensource.py \
    --backbone         uniformerv2_k400 \
    --backbone_weights /path/to/k400+k710_uniformerv2_b16_8x224.pyth \
    --head_ckpt        ../checkpoints/pvp_heads/uniformerv2_k400_vpvp.pt \
    --data_root        /path/to/aigvdbench \
    --num_frames       8 \
    --batch_size       32

# Qwen3-VL-8B vision tower
python eval_opensource.py \
    --backbone         qwen3vl_8b \
    --backbone_weights /path/to/Qwen3-VL-8B-Instruct \
    --head_ckpt        ../checkpoints/pvp_heads/qwen3vl_8b_vpvp.pt \
    --data_root        /path/to/aigvdbench \
    --num_frames       8 \
    --batch_size       16
```

Each run prints overall + per-generator AUC, and (with `--out_json`) writes a
JSON file containing the full per-generator breakdown.

## Repo layout

```
.
├── README.md                    (this file)
├── requirements.txt
├── src/
│   ├── pvp_head.py              V-PVP head + VelocityGatedAttn
│   ├── dataset.py               Open-Source split frame loader
│   ├── eval_opensource.py       evaluation entry point
│   └── backbones/
│       ├── __init__.py          BACKBONE_REGISTRY
│       ├── base.py              DTPBackbone abstract class
│       ├── timesformer_k400.py
│       ├── videomae_k400.py
│       ├── uniformerv2_k400.py
│       ├── qwen3vl_8b.py
│       ├── vit.py / vit_utils.py / weights.py / build.py / features.py
│       ├── conv2d_same.py / linear.py
│       └── _vendored/uniformerv2/    (from the official UniFormerV2 repo)
└── checkpoints/
    ├── README.md
    └── pvp_heads/
        ├── timesformer_k400_vpvp.pt
        ├── uniformerv2_k400_vpvp.pt
        ├── videomae_k400_vpvp.pt
        └── qwen3vl_8b_vpvp.pt
```

## Method (one paragraph)

Given a pre-trained video transformer producing patch tokens `z ∈ ℝ^{B×P×T'×D}`,
V-PVP computes the velocity field `v[p,t] = z[p,t+1] - z[p,t]`. Two streams
are formed: a velocity-gated context

```
α[p,t] = softmax_p( MLP([ z[p,t], ‖v[p,t]‖₂ ]) )
c̃[t]  = Σ_p α[p,t] · z[p,t]
```

and a raw per-channel magnitude `r[t] = (1/P) Σ_p |v[p,t]|`. Both are linearly
projected to `H/2` channels each, concatenated, passed through a temporal
Conv1D (k=3), mean-pooled across `t`, and read out by a single linear layer
into a logit. The total trainable head has ~0.5 M parameters; the backbone is
frozen.
