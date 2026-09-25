"""Frozen VideoMAE-B/16 (Kinetics-400 fine-tuned) backbone wrapper.

Uses the HuggingFace ``transformers`` implementation. The K400 checkpoint
expects 16 frames of 224x224 input and uses ``tubelet_size=2``, so the output
temporal length is T' = 16 / 2 = 8 and the V-PVP head sees 7 velocity steps.

The encoder returns (B, 1568, 768) with 1568 = 8 x 196 (time x spatial); this
wrapper reshapes it to (B, P=196, T'=8, D=768).

Details:
  * Input tensor order is (B, T, C, H, W); the wrapper transposes internally.
  * Normalization is ImageNet standard (mean 0.485/0.456/0.406,
    std 0.229/0.224/0.225).
  * The K400 checkpoint has ``use_mean_pooling=True``, so ``VideoMAEModel``
    applies no final LayerNorm and ``last_hidden_state`` is the output of the
    last transformer block before any final normalization (the token tap used
    in the paper). This is checked at construction time.
  * The original checkpoint stores the attention biases as ``q_bias`` /
    ``v_bias`` (no key bias). Depending on the ``transformers`` version the
    model either keeps that layout or expects ``query.bias`` / ``key.bias`` /
    ``value.bias``; the weights are remapped to whichever layout the installed
    version uses, so nothing is left randomly initialized.

Checkpoint: ``MCG-NJU/videomae-base-finetuned-kinetics`` on HuggingFace,
downloaded as a folder (config.json, preprocessor_config.json,
model.safetensors).
"""
import os

import torch

from .base import FrozenVideoBackbone


_NATIVE_T = 16    # input frames; tubelet_size=2 halves this to T'=8
_TUBELET = 2
_SPATIAL_P = 196  # 14 x 14


class VideoMAEK400(FrozenVideoBackbone):
    """Frozen VideoMAE-B/16 fine-tuned on Kinetics-400. 12 transformer blocks."""
    max_extract_layer = 11  # 12 encoder layers, 0-indexed

    def __init__(self, pretrained_model, num_frames=16, img_size=224, patch_size=16):
        super().__init__()
        if img_size != 224:
            raise ValueError(f"VideoMAE-B/16 K400 trained at 224; got {img_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_spatial = _SPATIAL_P
        self.embed_dim = 768
        # VideoMAE preprocessor config uses ImageNet standard statistics.
        self.input_mean = (0.485, 0.456, 0.406)
        self.input_std  = (0.229, 0.224, 0.225)

        from transformers import VideoMAEModel, VideoMAEConfig
        # Build the bare encoder (no classification head) from the config and
        # load the remapped weights ourselves.
        cfg = VideoMAEConfig.from_pretrained(pretrained_model)
        self.model = VideoMAEModel(cfg)
        if getattr(self.model, "layernorm", None) is not None:
            raise RuntimeError(
                "expected use_mean_pooling=True (no final LayerNorm) so that "
                "last_hidden_state is the pre-final-norm block output")
        self._load_weights(pretrained_model)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    def _load_weights(self, pretrained_model):
        """Load the checkpoint into the bare encoder, remapping the attention
        biases (``q_bias`` / ``v_bias``) to the layout of the installed
        ``transformers`` version."""
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

        # Older transformers keep separate q_bias / v_bias parameters; newer
        # versions fold them into query/key/value Linear biases.
        model_keys = set(self.model.state_dict().keys())
        legacy_layout = any(k.endswith('.q_bias') for k in model_keys)

        fixed = {}
        for k, v in sd.items():
            # The classification checkpoint stores keys under "videomae.*".
            nk = k[len('videomae.'):] if k.startswith('videomae.') else k
            if not legacy_layout:
                if nk.endswith('.q_bias'):
                    nk = nk.replace('.q_bias', '.query.bias')
                elif nk.endswith('.v_bias'):
                    nk = nk.replace('.v_bias', '.value.bias')
            fixed[nk] = v
        if not legacy_layout:
            # The original model has no key bias; zero-init it where expected.
            for k in list(fixed):
                if k.endswith('.query.bias'):
                    kb = k.replace('.query.bias', '.key.bias')
                    fixed.setdefault(kb, torch.zeros_like(fixed[k]))

        missing, unexpected = self.model.load_state_dict(fixed, strict=False)
        # classifier / fc_norm are unexpected because we use the bare encoder.
        real_missing = [k for k in missing if not k.startswith('pooler')]
        if real_missing:
            raise RuntimeError(f"VideoMAE load: missing keys: {real_missing[:5]}")

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
        """x: (B, C, T, H, W) -> (B, P=196, T'=8, D=768)."""
        if layer is not None and (layer < 0 or layer > self.max_extract_layer):
            raise ValueError(
                f"layer {layer} out of range [0, {self.max_extract_layer}]"
            )

        # Resample T -> 16, then transpose to (B, T, C, H, W).
        x16 = self._temporal_resample(x, _NATIVE_T)
        x16 = x16.permute(0, 2, 1, 3, 4).contiguous()

        if layer is None:
            out = self.model(x16)
            hid = out.last_hidden_state       # (B, 1568, D)
        else:
            out = self.model(x16, output_hidden_states=True)
            # hidden_states[0] = post-embedding, hidden_states[i+1] = after block i.
            hid = out.hidden_states[layer + 1]

        B, N, D = hid.shape
        T_out = _NATIVE_T // _TUBELET       # 8
        # VideoMAE flattens tokens as (T_out, P_spatial); reshape accordingly.
        feat = hid.view(B, T_out, _SPATIAL_P, D).permute(0, 2, 1, 3).contiguous()
        return feat                          # (B, P=196, T'=8, D)
