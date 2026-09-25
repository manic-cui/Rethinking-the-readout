# V-PVP: Velocity-Gated Patch Velocity Profiling

Code release for **V-PVP**, a lightweight readout for AI-generated video
detection on top of a **frozen** video backbone. Given patch tokens
`z ∈ ℝ^{B×P×T'×D}` from a pre-trained video transformer, V-PVP computes the
patch velocity field `v[t] = z[t+1] - z[t]`, aggregates the patches of each
frame pair with a velocity-gated attention to get `z̃[t]`, keeps a
channel-faithful velocity-magnitude stream `r[t] = (1/P) Σ_p |v[t,p]|`, and
reads the two streams out with a small temporal Conv1D + linear classifier.
The head has ~0.5 M trainable parameters; the backbone is never updated.

This release contains the VideoMAE-Base configuration used for the main
results:

- `src/` — V-PVP head, frozen VideoMAE-K400 backbone wrapper, AIGVDBench
  Open-Source test loader, evaluation entry point
- `checkpoints/pvp_heads/` — trained V-PVP head weights

## Released checkpoints

| File | Backbone | Protocol | mga AUC |
| ---- | -------- | -------- | ------: |
| `pvp_heads/videomae_k400_vpvp.pt` | VideoMAE-B/16 K400, frozen | AIGVDBench, 20-generator Open-Source test split | **0.9528** |
| `pvp_heads/genvidbench_videomae_auc0.9375.pt` | VideoMAE-B/16 K400, frozen | GenVidBench-143k cross-source protocol | **0.9375** |

Both heads are `state_dict`s of `PVPHead("velocity_gated", D=768)`
(`src/pvp_head.py`, 492,674 parameters) trained with the backbone fully
frozen (no LoRA, no fine-tuning). `eval_opensource.py` implements the
AIGVDBench Open-Source evaluation (3000 real + 60000 fake = 63000 videos);
see `checkpoints/README.md` for the GenVidBench head.

## Install

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.13, `torch` 2.x and `transformers` 4.x / 5.x. The
VideoMAE wrapper loads the checkpoint itself and remaps the attention biases
to the layout of the installed `transformers` version, so both major versions
give identical features.

## Backbone weights

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

## Data

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

## Run evaluation

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

## Repo layout

```
.
├── README.md                    (this file)
├── requirements.txt
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

## Method (one paragraph)

Given a pre-trained video transformer producing patch tokens
`z ∈ ℝ^{B×P×T'×D}`, V-PVP computes the velocity field
`v[p,t] = z[p,t+1] - z[p,t]`. Two streams are formed: a velocity-gated
aggregation

```
α[p,t] = softmax_p( MLP([ z[p,t], ‖v[p,t]‖₂ ]) )
z̃[t]  = Σ_p α[p,t] · z[p,t]
```

and a channel-faithful magnitude `r[t] = (1/P) Σ_p |v[p,t]|`. Both are
linearly projected to `H/2 = 128` channels each, concatenated, passed through
GELU → Conv1D (k=3) → GELU → mean over `t` → Dropout(0.3) → Linear, giving
one logit per clip. The head has 492,674 trainable parameters; the backbone
stays frozen.
