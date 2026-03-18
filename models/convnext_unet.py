"""ConvNeXt Encoder + U-Net (segmentation) + Bottleneck-based Classification.

This module is intentionally self-contained.

Forward API:
- If x_trans is provided:
    - return_cls=False: (seg_long, seg_trans)
    - return_cls=True:  (seg_long, seg_trans, cls_logits)
- If x_trans is None:
    - return_cls=False: seg_long
    - return_cls=True:  (seg_long, cls_logits)
"""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as e:
    raise ImportError("timm is required for ConvNeXt. Please `pip install timm`.") from e


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt encoder returning multi-scale feature maps.

    Returns:
        c0: stem feature at H/2
        c1..c4: timm features at H/4, H/8, H/16, H/32
    """

    def __init__(self, in_chans: int = 1, model_name: str = "convnext_nano", pretrained: bool = True):
        super().__init__()

        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        feat_channels = self.feature_extractor.feature_info.channels()
        self.c1_ch, self.c2_ch, self.c3_ch, self.c4_ch = feat_channels

        # ConvNeXt stem starts at H/4; add shallow stem for H/2 skip.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, self.c1_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.c1_ch),
            nn.ReLU(inplace=True),
        )

        self.out_channels = (self.c1_ch, self.c1_ch, self.c2_ch, self.c3_ch, self.c4_ch)

    def forward(self, x: torch.Tensor):
        c0 = self.stem(x)
        c1, c2, c3, c4 = self.feature_extractor(x)
        return c0, c1, c2, c3, c4

    def freeze_stages(self, stages: List[int], freeze_stem: bool = True):
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

        if not stages:
            return

        # timm ConvNeXt generally uses 'stages.{i}' and 'downsample_layers.{i}'.
        for name, param in self.feature_extractor.named_parameters():
            lname = name.lower()
            for s in stages:
                if f"stages.{s}" in lname or f"downsample_layers.{s}" in lname:
                    param.requires_grad = False
                    break

    def unfreeze_all(self):
        for p in self.stem.parameters():
            p.requires_grad = True
        for p in self.feature_extractor.parameters():
            p.requires_grad = True


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        c0_ch: int,
        c1_ch: int,
        c2_ch: int,
        c3_ch: int,
        c4_ch: int,
        num_seg_classes: int = 3,
    ):
        super().__init__()
        out0_ch = max(32, c0_ch // 2)

        self.up_43 = UpBlock(in_ch=c4_ch, skip_ch=c3_ch, out_ch=c3_ch)
        self.up_32 = UpBlock(in_ch=c3_ch, skip_ch=c2_ch, out_ch=c2_ch)
        self.up_21 = UpBlock(in_ch=c2_ch, skip_ch=c1_ch, out_ch=c1_ch)
        self.up_10 = UpBlock(in_ch=c1_ch, skip_ch=c0_ch, out_ch=out0_ch)
        self.final_head = nn.Conv2d(out0_ch, num_seg_classes, kernel_size=1)

    def forward(self, c0, c1, c2, c3, c4):
        d3 = self.up_43(c4, c3)
        d2 = self.up_32(d3, c2)
        d1 = self.up_21(d2, c1)
        d0 = self.up_10(d1, c0)
        return self.final_head(d0)


class MorphologicalFeatureExtractor(nn.Module):
    """Extract simple structural features from predicted segmentation."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.plaque_class = 1
        self.vessel_class = 2
        self.out_dim = 10

    def forward(self, seg_logits: torch.Tensor, seg_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if seg_probs is None:
            seg_probs = F.softmax(seg_logits, dim=1)

        batch_size = seg_logits.shape[0]
        device = seg_logits.device
        features_list: List[List[float]] = []

        seg_pred = torch.argmax(seg_logits, dim=1)
        plaque_mask = (seg_pred == self.plaque_class).float()
        vessel_mask = (seg_pred == self.vessel_class).float()
        foreground_mask = (seg_pred > 0).float()

        for b in range(batch_size):
            plaque_pixels = plaque_mask[b]
            vessel_pixels = vessel_mask[b]
            fg_pixels = foreground_mask[b]

            plaque_area = plaque_pixels.sum().item()
            vessel_area = vessel_pixels.sum().item()
            plaque_ratio = plaque_area / (vessel_area + 1e-6)

            fg_area = fg_pixels.sum().item()
            plaque_ratio_fg = plaque_area / (fg_area + 1e-6)

            plaque_conf = seg_probs[b, self.plaque_class] * plaque_mask[b]
            plaque_conf_weighted = plaque_conf.sum().item()

            if plaque_area > 0:
                plaque_coords = torch.nonzero(plaque_pixels, as_tuple=True)
                if len(plaque_coords[0]) > 0:
                    height_range = (plaque_coords[0].max() - plaque_coords[0].min()).float().item()
                    width_range = (plaque_coords[1].max() - plaque_coords[1].min()).float().item()
                    plaque_perimeter_approx = 2 * (height_range + width_range)
                else:
                    height_range = 0.0
                    width_range = 0.0
                    plaque_perimeter_approx = 0.0
            else:
                height_range = 0.0
                width_range = 0.0
                plaque_perimeter_approx = 0.0

            vessel_conf = seg_probs[b, self.vessel_class] * vessel_mask[b]
            vessel_conf_weighted = vessel_conf.sum().item()

            if plaque_area > 0:
                mean_plaque_conf = (seg_probs[b, self.plaque_class] * plaque_mask[b]).sum().item() / (plaque_area + 1e-6)
            else:
                mean_plaque_conf = 0.0

            fg_ratio = fg_area / (seg_logits.shape[2] * seg_logits.shape[3])

            batch_features = [
                plaque_ratio,
                plaque_ratio_fg,
                plaque_conf_weighted,
                height_range / seg_logits.shape[2],
                width_range / seg_logits.shape[3],
                plaque_perimeter_approx / (seg_logits.shape[2] + seg_logits.shape[3]),
                vessel_conf_weighted,
                mean_plaque_conf,
                fg_ratio,
                float(plaque_area) / (seg_logits.shape[2] * seg_logits.shape[3]),
            ]
            features_list.append(batch_features)

        return torch.tensor(features_list, dtype=torch.float32, device=device)


class ConvNeXtUNet(nn.Module):
    """ConvNeXt + U-Net (Two-view segmentation) + bottleneck + morphological classification."""

    def __init__(
        self,
        in_chans: int = 1,
        num_seg_classes: int = 3,
        cls_class_num: int = 1,
        pretrained_encoder: bool = True,
        convnext_model: str = "convnext_nano",
        cls_hidden: int = 512,
        cls_dropout: float = 0.3,
        plaque_gate_alpha: float = 1.0,
        morph_proj_dim: int = 64,
    ):
        super().__init__()

        self.encoder = ConvNeXtEncoder(in_chans=in_chans, model_name=convnext_model, pretrained=pretrained_encoder)
        c0_ch, c1_ch, c2_ch, c3_ch, c4_ch = self.encoder.out_channels

        self.decoder_long = UNetDecoder(c0_ch, c1_ch, c2_ch, c3_ch, c4_ch, num_seg_classes=num_seg_classes)
        self.decoder_trans = UNetDecoder(c0_ch, c1_ch, c2_ch, c3_ch, c4_ch, num_seg_classes=num_seg_classes)

        self.morph_extractor = MorphologicalFeatureExtractor(num_classes=num_seg_classes)
        num_morph_features = self.morph_extractor.out_dim
        self.morph_proj = nn.Sequential(
            nn.LayerNorm(num_morph_features),
            nn.Linear(num_morph_features, morph_proj_dim),
            nn.ReLU(inplace=True),
        )

        bottleneck_dim = c4_ch
        cls_input_dim = bottleneck_dim * 2 + morph_proj_dim * 2
        self.cls_head = nn.Sequential(
            nn.Linear(cls_input_dim, cls_hidden),
            nn.LayerNorm(cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, cls_hidden // 2),
            nn.LayerNorm(cls_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden // 2, cls_class_num),
        )

        self.plaque_class = 1
        self.plaque_gate_alpha = plaque_gate_alpha

    def freeze_encoder(self, stages: List[int], freeze_stem: bool = True):
        self.encoder.freeze_stages(stages=stages, freeze_stem=freeze_stem)

    def unfreeze_encoder(self):
        self.encoder.unfreeze_all()

    def _mask_weighted_pool(self, feat_map: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        seg_probs = F.softmax(seg_logits.detach(), dim=1)
        p_plaque = seg_probs[:, self.plaque_class : self.plaque_class + 1]
        p_plaque = F.interpolate(p_plaque, size=feat_map.shape[2:], mode="bilinear", align_corners=False)
        weights = 1.0 + self.plaque_gate_alpha * p_plaque
        weighted = feat_map * weights
        sum_feat = weighted.sum(dim=(2, 3))
        sum_w = weights.sum(dim=(2, 3)).clamp_min(1e-6)
        return sum_feat / sum_w

    def forward(
        self,
        x_long: torch.Tensor,
        x_trans: Optional[torch.Tensor] = None,
        return_cls: bool = False,
        # API-compatibility with other training code; ignored here.
        return_moe_aux: bool = False,
        router_temp: float = 1.0,
        router_noise_std: float = 0.0,
        moe_lb_lambda: float = 0.0,
    ):
        _ = (return_moe_aux, router_temp, router_noise_std, moe_lb_lambda)

        H, W = x_long.shape[2], x_long.shape[3]

        c0_l, c1_l, c2_l, c3_l, c4_l = self.encoder(x_long)
        seg_long = self.decoder_long(c0_l, c1_l, c2_l, c3_l, c4_l)
        seg_long = F.interpolate(seg_long, size=(H, W), mode="bilinear", align_corners=False)

        if x_trans is None:
            if return_cls:
                seg_long_detached = seg_long.detach()
                probs_long = F.softmax(seg_long_detached, dim=1)

                emb_long = self._mask_weighted_pool(c4_l, seg_long_detached)
                emb_trans = emb_long.clone()

                feat_morph_long = self.morph_extractor(seg_long_detached, probs_long)
                feat_morph_trans = feat_morph_long.clone()
                feat_morph_long = self.morph_proj(feat_morph_long)
                feat_morph_trans = self.morph_proj(feat_morph_trans)

                feat_cls = torch.cat([emb_long, emb_trans, feat_morph_long, feat_morph_trans], dim=1)
                cls_logits = self.cls_head(feat_cls)
                return seg_long, cls_logits
            return seg_long

        c0_t, c1_t, c2_t, c3_t, c4_t = self.encoder(x_trans)
        seg_trans = self.decoder_trans(c0_t, c1_t, c2_t, c3_t, c4_t)
        seg_trans = F.interpolate(seg_trans, size=(H, W), mode="bilinear", align_corners=False)

        if return_cls:
            seg_long_detached = seg_long.detach()
            seg_trans_detached = seg_trans.detach()
            probs_long = F.softmax(seg_long_detached, dim=1)
            probs_trans = F.softmax(seg_trans_detached, dim=1)

            emb_long = self._mask_weighted_pool(c4_l, seg_long_detached)
            emb_trans = self._mask_weighted_pool(c4_t, seg_trans_detached)

            feat_morph_long = self.morph_extractor(seg_long_detached, probs_long)
            feat_morph_trans = self.morph_extractor(seg_trans_detached, probs_trans)
            feat_morph_long = self.morph_proj(feat_morph_long)
            feat_morph_trans = self.morph_proj(feat_morph_trans)

            feat_cls = torch.cat([emb_long, emb_trans, feat_morph_long, feat_morph_trans], dim=1)
            cls_logits = self.cls_head(feat_cls)
            return seg_long, seg_trans, cls_logits

        return seg_long, seg_trans
