"""V-PVP head: dual-stream, velocity-gated readout on frozen patch tokens.

Given backbone patch features ``z`` of shape ``(B, P, T', D)``:
  v[t]      = z[t+1] - z[t]                       # (B, P, T'-1, D)
  c_tilde   = sum_p alpha_t^(p) * z[:, p, :T'-1]  # velocity-gated context
              with alpha = softmax_p(MLP([z, ||v||_2]))
  r[t]      = (1/P) * sum_p |v[t, p]|              # raw per-channel magnitude
  X[t]      = concat([proj_z(c_tilde), proj_r(r)], dim=channel)
            -> Conv1d(k=3) -> mean over t -> Linear -> logit

This is the "velocity_gated" head used for all V-PVP results in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityGatedAttn(nn.Module):
    """Spatial attention over patches, gated by per-patch velocity magnitude.

    The score for patch p at time t is computed by a small MLP that takes the
    concatenation of ``z[p, t]`` and the scalar ``||v[p, t]||_2``. Softmax over
    P yields content-and-motion-aware weights.
    """
    def __init__(self, D, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(D + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_paired, v_norm):
        # z_paired: (B, P, T-1, D), v_norm: (B, P, T-1)
        x = torch.cat([z_paired, v_norm.unsqueeze(-1)], dim=-1)
        score = self.mlp(x).squeeze(-1)
        attn = torch.softmax(score, dim=1)
        return (attn.unsqueeze(-1) * z_paired).sum(dim=1)   # (B, T-1, D)


class PVPHead(nn.Module):
    """Dual-stream V-PVP head.

    Args:
        agg_name: ``"velocity_gated"`` (V-PVP) or ``"plain_attn"`` (ablation).
        D:        embedding dim of the frozen backbone.
        hidden:   internal channel width (must be even; split between z and r
                  streams, hidden/2 each).
        drop:     dropout before the final linear.
    """
    def __init__(self, agg_name, D, hidden=256, drop=0.3):
        super().__init__()
        assert hidden % 2 == 0
        if agg_name == "velocity_gated":
            self.agg = VelocityGatedAttn(D=D)
        else:
            raise ValueError(f"unknown agg_name: {agg_name}")
        self.proj_z = nn.Linear(D, hidden // 2)
        self.proj_r = nn.Linear(D, hidden // 2)
        self.conv   = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.fc     = nn.Linear(hidden, 1)
        self.drop   = nn.Dropout(drop)

    def forward(self, z, v):
        # z: (B, P, T', D), v: (B, P, T'-1, D)
        z_paired = z[:, :, :-1, :]
        v_norm   = v.norm(dim=-1)
        c_tilde  = self.agg(z_paired, v_norm)              # (B, T-1, D)
        r        = v.abs().mean(dim=1)                     # (B, T-1, D)
        h        = torch.cat([self.proj_z(c_tilde),
                              self.proj_r(r)], dim=-1)     # (B, T-1, H)
        h = F.gelu(h).transpose(1, 2)                      # (B, H, T-1)
        h = F.gelu(self.conv(h)).mean(dim=2)               # (B, H)
        return self.fc(self.drop(h)).squeeze(-1)           # (B,)
