# V-PVP head checkpoints

Each `.pt` file is a `state_dict` of `PVPHead("velocity_gated", D=768)` from
`src/pvp_head.py`, trained on top of a **frozen** VideoMAE-B/16 backbone
fine-tuned on Kinetics-400 (`MCG-NJU/videomae-base-finetuned-kinetics`).
The head has 492,674 trainable parameters.

| File | Training / evaluation protocol | mga AUC |
| ---- | ------------------------------ | ------: |
| `pvp_heads/videomae_k400_vpvp.pt` | AIGVDBench: train on Real + Open-Sora + CogVideoX1.5 + EasyAnimate, evaluate on the 20-generator Open-Source test split | 0.9528 |
| `pvp_heads/genvidbench_videomae_auc0.9375.pt` | GenVidBench-143k cross-source protocol (train: Pika, VideoCrafterV2, ModelScope, T2V-Zero; test: MuseV, SVD, CogVideo, Mora) | 0.9375 |

The VideoMAE **backbone** weights are **not** included; download them from
HuggingFace (see the project README). `src/eval_opensource.py` implements the
AIGVDBench Open-Source evaluation; a GenVidBench data loader is not part of
this release, but the GenVidBench head can be applied to any
`(B, P, T', D)` VideoMAE patch tokens the same way.

To load a checkpoint manually:

```python
import torch
from src.pvp_head import PVPHead

head = PVPHead("velocity_gated", D=768)
head.load_state_dict(torch.load("checkpoints/pvp_heads/videomae_k400_vpvp.pt",
                                map_location="cpu"))
head.eval()
```
