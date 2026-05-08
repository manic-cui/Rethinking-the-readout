"""VideoMAE K400 backbone wrapper (HuggingFace transformers).

VideoMAE ViT-B/16 fine-tuned on Kinetics-400 (MCG-NJU release). Expects
16 frames of 224×224 input and uses tubelet_size=2, so output temporal dim
is T' = 16/2 = 8 — same trajectory length as TimeSformer / SlowFast (slow).

Patch token output: (B, 1568, D=768) where 1568 = 8 × 196 (time × spatial).
We reshape to (B, P=196, T'=8, D=768).

Key details:
  * Input tensor order is (B, T, C, H, W) — transposed inside the wrapper.
  * Normalization is **ImageNet standard**, not (0.45, 0.45, 0.45) like the
    TimeSformer/SlowFast/X3D K400 checkpoints.
  * The HF checkpoint stores q_bias/v_bias (no k_bias) from original VideoMAE;
    HF's ViT expects separate query/key/value biases. We remap state_dict keys
    at load time (q_bias → query.bias, v_bias → value.bias, zero key.bias).

Checkpoint source: `MCG-NJU/videomae-base-finetuned-kinetics` on HuggingFace
(downloaded to DTP/weights/videomae_k400/).
"""
import torch

from .base import DTPBackbone


_NATIVE_T = 16  # T_in. tubelet=2 halves to T_out=8.
_TUBELET = 2
_SPATIAL_P = 196  # 14x14


class VideoMAEK400(DTPBackbone):
    """Frozen VideoMAE-B/16 fine-tuned on Kinetics-400. 12 transformer blocks."""
    max_extract_layer = 11  # 12 encoder layers, 0-indexed

    def __init__(self, pretrained_model, num_frames=8, img_size=224, patch_size=16):
        super().__init__()
        if img_size != 224:
            raise ValueError(f"VideoMAE-B/16 K400 trained at 224; got {img_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_spatial = _SPATIAL_P
        self.embed_dim = 768
        # VideoMAE preprocessor config uses ImageNet standard.
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std  = (0.229, 0.224, 0.225)

        from transformers import VideoMAEModel, VideoMAEConfig
        # Build bare encoder (no classification head) from config, then load
        # our remapped weights ourselves — bypasses HF's noisy qkv_bias warning.
        cfg = VideoMAEConfig.from_pretrained(pretrained_model)
        self.model = VideoMAEModel(cfg)
        self._remap_qkv_bias(pretrained_model)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _remap_qkv_bias(self, pretrained_model):
        """Rename checkpoint's q_bias/v_bias to match HF's query.bias/value.bias.

        Original VideoMAE omits k_bias; HF's ViT expects one. We zero-init it.
        """
        import os, glob
        st_path = None
        for cand in ('model.safetensors', 'pytorch_model.bin'):
            p = os.path.join(pretrained_model, cand)
            if os.path.isfile(p):
                st_path = p
                break
        if st_path is None:
            raise FileNotFoundError(f"No weights file in {pretrained_model}")

        if st_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            sd = load_file(st_path)
        else:
            sd = torch.load(st_path, map_location='cpu', weights_only=False)

        fixed = {}
        # Strip the "videomae." prefix that the classification checkpoint stores,
        # since we instantiated VideoMAEModel directly (no videomae.* prefix).
        for k, v in sd.items():
            nk = k[len('videomae.'):] if k.startswith('videomae.') else k
            if nk.endswith('.q_bias'):
                nk = nk.replace('.q_bias', '.query.bias')
            elif nk.endswith('.v_bias'):
                nk = nk.replace('.v_bias', '.value.bias')
            fixed[nk] = v
        # Create zero key.bias entries wherever query.bias exists.
        for k in list(fixed):
            if k.endswith('.query.bias'):
                kb = k.replace('.query.bias', '.key.bias')
                fixed.setdefault(kb, torch.zeros_like(fixed[k]))

        missing, unexpected = self.model.load_state_dict(fixed, strict=False)
        # classifier/fc_norm are unexpected because we're using the bare encoder
        # — that's expected. Anything else is a real issue.
        real_missing = [k for k in missing if not k.startswith('pooler')]
        if real_missing:
            raise RuntimeError(f"VideoMAE load: unexpected missing keys: {real_missing[:5]}")

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _temporal_resample(x, target_T):
        T = x.shape[2]
        if T == target_T:
            return x
        idx = torch.linspace(0, T - 1, target_T, device=x.device).round().long()
        return x.index_select(2, idx)

    @torch.no_grad()
    def extract_patch_features(self, x, layer=None):
        """x: (B, C, T, H, W) → (B, P=196, T'=8, D=768)."""
        if layer is not None and (layer < 0 or layer > self.max_extract_layer):
            raise ValueError(
                f"layer {layer} out of range [0, {self.max_extract_layer}]"
            )

        # Resample T → 16 then transpose to (B, T, C, H, W).
        x16 = self._temporal_resample(x, _NATIVE_T)
        x16 = x16.permute(0, 2, 1, 3, 4).contiguous()

        if layer is None:
            out = self.model(x16)
            hid = out.last_hidden_state       # (B, 1568, D)
        else:
            out = self.model(x16, output_hidden_states=True)
            # hidden_states[0] = post-embedding, hidden_states[i+1] = after layer i.
            hid = out.hidden_states[layer + 1]

        B, N, D = hid.shape
        T_out = _NATIVE_T // _TUBELET       # 8
        # VideoMAE flattens as (T_out, P_spatial); reshape accordingly.
        feat = hid.view(B, T_out, _SPATIAL_P, D).permute(0, 2, 1, 3).contiguous()
        return feat                          # (B, P=196, T'=8, D)
