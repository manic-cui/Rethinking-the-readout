# V-PVP head checkpoints

This folder contains the trained V-PVP head weights for the four ViT
backbones reported in the paper. Each `.pt` file is a `state_dict` of the
`PVPHead("velocity_gated", D=...)` module defined in `src/pvp_head.py`.

| File                           | Backbone           | mga AUC (Open-Source) |
| ------------------------------ | ------------------ | --------------------: |
| `pvp_heads/timesformer_k400_vpvp.pt` | TimeSformer-Base K400 |              0.9124   |
| `pvp_heads/uniformerv2_k400_vpvp.pt` | UniFormerV2-Base K400 |              0.9356   |
| `pvp_heads/videomae_k400_vpvp.pt`    | VideoMAE-Base K400    |              0.9528   |
| `pvp_heads/qwen3vl_8b_vpvp.pt`       | Qwen3-VL-8B (vision)  |              0.9778   |

The K400 / vision-tower **backbone** weights are **not** included — please
download them from the original public sources (see the project README).
V-PVP only trains the head; the backbone is frozen.

To load a checkpoint manually:

```python
import torch
from src.pvp_head import PVPHead

# D is 1024 for qwen3vl_8b, 768 for the other three
head = PVPHead("velocity_gated", D=768)
head.load_state_dict(torch.load("checkpoints/pvp_heads/videomae_k400_vpvp.pt",
                                map_location="cpu"))
head.eval()
```
