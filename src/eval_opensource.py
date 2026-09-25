"""Evaluate a trained V-PVP head on the AIGVDBench Open-Source split.

Computes per-generator AUC, mean-of-generators AUC (mga AUC), and overall
AUC on the test set. Each backbone is loaded once (frozen, fp16 autocast),
patch features ``z`` of shape ``(B, P, T', D)`` are extracted, the velocity
field ``v = z[:, :, 1:] - z[:, :, :-1]`` is computed, and the head produces
a scalar logit per video.

Usage:
    python -m src.eval_opensource \
        --backbone videomae_k400 \
        --backbone_weights <path-or-hub-id> \
        --head_ckpt    checkpoints/pvp_heads/videomae_k400_vpvp.pt \
        --data_root    <path-to-aigvdbench> \
        --num_frames   16 \
        --batch_size   32 \
        --out_json     results/videomae_open_source.json

Expected mga AUC on the Open-Source split with the released checkpoint
(20 generators, 3000 real + 60000 fake videos):

    videomae_k400     ->  0.9528   (V-PVP row of Table 1)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbones import BACKBONE_REGISTRY
from dataset import OpenSourceConfig, OpenSourceTest, collate
from pvp_head import PVPHead


# Frame count the released checkpoint was trained at.
DEFAULT_NUM_FRAMES = {
    "videomae_k400": 16,
}

# Dataset-side normalization applied by ``OpenSourceTest`` (mean 0.45 /
# std 0.225). The eval loop undoes it and re-applies the backbone's own
# statistics below; this is the exact preprocessing path the released
# checkpoint was trained and evaluated with.
DS_MEAN = (0.45, 0.45, 0.45)
DS_STD  = (0.225, 0.225, 0.225)


def renormalize(clips, ds_mean, ds_std, bb_mean, bb_std):
    """Undo dataset normalization, re-apply backbone's. clips: (B, C, T, H, W)."""
    x = clips * ds_std + ds_mean
    return (x - bb_mean) / bb_std


def compute_metrics(scores, labels, cats, threshold=0.5):
    """mga AUC = mean-of-generators AUC over the per-(real, generator) pools."""
    cats   = np.asarray(cats)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    real   = (labels == 0)
    preds  = (scores > threshold).astype(int)

    auc_per_gen, acc_per_gen = {}, {}
    for c in sorted(set(cats[~real].tolist())):
        gen_mask = (cats == c) & (labels == 1)
        if gen_mask.sum() < 2 or real.sum() < 2:
            continue
        y = np.r_[np.zeros(real.sum()), np.ones(gen_mask.sum())]
        s = np.r_[scores[real],          scores[gen_mask]]
        p = np.r_[preds[real],           preds[gen_mask]]
        auc_per_gen[c] = float(roc_auc_score(y, s))
        acc_per_gen[c] = float((p == y).mean())

    return {
        "auc_per_gen": auc_per_gen,
        "acc_per_gen": acc_per_gen,
        "mga_auc":     float(np.mean(list(auc_per_gen.values()))),
        "mga_acc":     float(np.mean(list(acc_per_gen.values()))),
        "overall_auc": float(roc_auc_score(labels, scores)),
        "overall_acc": float((preds == labels).mean()),
        "n_real":      int(real.sum()),
        "n_fake":      int((~real).sum()),
    }


@torch.no_grad()
def evaluate(backbone, head, loader, device, ds_norm, bb_norm):
    head.eval()
    backbone.eval()
    scores_all, labels_all, cats_all = [], [], []
    n_done = 0
    t0 = time.time()
    for clips, labels, cats in loader:
        clips = clips.to(device, non_blocking=True)
        x = renormalize(clips, *ds_norm, *bb_norm)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = backbone.extract_patch_features(x)
        z = z.to(torch.float32)
        if z.size(2) < 2:
            raise RuntimeError(f"backbone returned T'={z.size(2)} < 2; need >=2 frames")
        v = z[:, :, 1:, :] - z[:, :, :-1, :]
        logit = head(z, v)
        scores_all.append(torch.sigmoid(logit).cpu().numpy())
        labels_all.append(labels.numpy())
        cats_all.extend(cats)
        n_done += clips.size(0)
        if n_done % 1024 == 0:
            print(f"  [{n_done}] {n_done/(time.time()-t0):.1f} v/s", flush=True)
    return (np.concatenate(scores_all),
            np.concatenate(labels_all),
            np.array(cats_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True,
                    choices=list(BACKBONE_REGISTRY.keys()))
    ap.add_argument("--backbone_weights", required=True,
                    help="path (or HuggingFace repo id) of the K400 backbone weights")
    ap.add_argument("--head_ckpt", required=True,
                    help="path to the V-PVP head state_dict (.pt)")
    ap.add_argument("--data_root", required=True,
                    help="AIGVDBench root containing Real/ and OpenSource/ folders")
    ap.add_argument("--num_frames", type=int, default=0,
                    help="0 -> backbone default (16 for VideoMAE)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default="",
                    help="if set, write per-generator AUC + summary to this JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nf = args.num_frames or DEFAULT_NUM_FRAMES[args.backbone]
    print(f"backbone={args.backbone}  num_frames={nf}  bs={args.batch_size}", flush=True)
    print(f"backbone_weights={args.backbone_weights}", flush=True)
    print(f"head_ckpt={args.head_ckpt}", flush=True)

    # ---- backbone (frozen) ----
    bb_cls   = BACKBONE_REGISTRY[args.backbone]
    backbone = bb_cls(pretrained_model=args.backbone_weights,
                      num_frames=nf, img_size=224).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    D = backbone.embed_dim
    print(f"  P={backbone.num_patches_spatial}  D={D}  "
          f"input_mean={backbone.input_mean}  input_std={backbone.input_std}", flush=True)

    # ---- head ----
    head = PVPHead("velocity_gated", D=D).to(device)
    sd = torch.load(args.head_ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    head.load_state_dict(sd)
    head.eval()
    n_head = sum(p.numel() for p in head.parameters())
    print(f"  head params: {n_head:,}", flush=True)

    # ---- data ----
    cfg = OpenSourceConfig(data_root=args.data_root, num_frames=nf,
                           crop_size=224, max_per_category=0)
    ds  = OpenSourceTest(cfg)
    print(f"  test N={len(ds)}", flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate)

    # ---- normalization tensors ----
    ds_mean = torch.tensor(DS_MEAN, device=device).view(1, 3, 1, 1, 1)
    ds_std  = torch.tensor(DS_STD,  device=device).view(1, 3, 1, 1, 1)
    bb_mean = torch.tensor(backbone.input_mean, device=device).view(1, 3, 1, 1, 1)
    bb_std  = torch.tensor(backbone.input_std,  device=device).view(1, 3, 1, 1, 1)

    # ---- run ----
    t0 = time.time()
    scores, labels, cats = evaluate(backbone, head, loader, device,
                                    (ds_mean, ds_std), (bb_mean, bb_std))
    elapsed = (time.time() - t0) / 60
    print(f"  inference done in {elapsed:.1f} min  ({len(scores)} videos)", flush=True)

    m = compute_metrics(scores, labels, cats, threshold=args.threshold)
    print(f"\nOpen-Source split results ({m['n_real']} real / {m['n_fake']} fake):")
    print(f"  mga AUC   = {m['mga_auc']:.4f}")
    print(f"  mga ACC   = {m['mga_acc']:.4f}")
    print(f"  overall AUC = {m['overall_auc']:.4f}")
    print(f"  overall ACC = {m['overall_acc']:.4f}")
    print("\nPer-generator AUC:")
    for c, a in m["auc_per_gen"].items():
        print(f"  {c:<32s} {a*100:6.2f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        out = {
            "backbone":   args.backbone,
            "num_frames": nf,
            "head_ckpt":  args.head_ckpt,
            **m,
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
