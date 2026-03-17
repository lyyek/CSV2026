import os

import torch
import torch.nn as nn


class SubmissionModel(nn.Module):
    """
    Wrapper for submission inference.

    Expected forward:
        seg_long_logits, seg_trans_logits, cls_logits = model(x_long, x_trans)

    Inputs:
        x_long:  [B, 1, H, W]
        x_trans: [B, 1, H, W]
    """

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core = core_model

    def forward(self, x_long: torch.Tensor, x_trans: torch.Tensor):
        outputs = self.core(x_long, x_trans, return_cls=True)

        if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise RuntimeError(
                "Submission model must return exactly 3 outputs: "
                "(seg_long_logits, seg_trans_logits, cls_logits)"
            )

        seg_long_logits, seg_trans_logits, cls_logits = outputs
        return seg_long_logits, seg_trans_logits, cls_logits


def build_submission_model():
    """
    Build the model used in inference.py.
    """
    from models.convnext_unet import ConvNeXtUNet

    core_model = ConvNeXtUNet(
        in_chans=1,
        num_seg_classes=3,
        cls_class_num=1,
        pretrained_seg=False,
        convnext_model="convnext_nano",
        cls_hidden=512,
        cls_dropout=0.3,
        plaque_gate_alpha=1.0,
        morph_proj_dim=64,
    )

    return SubmissionModel(core_model)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    return checkpoint


def load_weights_compat(model: nn.Module, weights_path: str, map_location="cpu"):
    """
    Load checkpoint weights with flexible handling of common checkpoint formats.
    Only parameters with matching names and shapes are loaded.
    """
    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights_path not found: {weights_path}")

    try:
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=map_location)
    except Exception:
        checkpoint = torch.load(weights_path, map_location=map_location, weights_only=False)

    state_dict = _extract_state_dict(checkpoint)

    target_model = model.core if hasattr(model, "core") else model
    current_state = target_model.state_dict()

    filtered_state = {}
    for key, value in state_dict.items():
        if key in current_state and hasattr(value, "shape") and value.shape == current_state[key].shape:
            filtered_state[key] = value

    target_model.load_state_dict(filtered_state, strict=False)
    return model