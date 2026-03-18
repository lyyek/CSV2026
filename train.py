import os
import sys
import json
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score
import segmentation_models_pytorch as smp
from medpy.metric.binary import dc

from utils import compute_nsd, DiceLoss, AverageMeter
from dataset.csv import CSVSemiDataset
from earlystopping import EarlyStopping
from models.convnext_unet import ConvNeXtUNet


# --------------------------------------------------------------------------
# 0. Default configuration
# --------------------------------------------------------------------------
DEFAULT_CFG = {
    # Data
    "data_root": "./CSV2026_Dataset_Train",
    "split_json": "./split4.json",

    # Model
    "model_type": "convnext_unet",
    "convnext_model": "convnext_nano",
    "pretrained_encoder": True,
    "plaque_gate_alpha": 1.0,
    "cls_hidden": 512,
    "cls_dropout": 0.3,
    "morph_proj_dim": 64,
    "img_size": 512,
    "num_classes": 3,
    "cls_class_num": 1,

    # Training
    "train_epochs": 150,
    "batch_size": 4,
    "learning_rate": 1e-4,
    "encoder_lr": 1e-4,
    "decoder_lr": 1e-4,
    "cls_head_lr": 1e-4,
    "weight_decay": 0.05,
    "gradient_accumulation_steps": 2,
    "cls_loss_weight": 0.3,
    "cls_warmup_epochs": 15,
    "cls_label_smoothing": 0.05,

    # Early stopping
    "early_stopping_patience": 30,

    # Checkpoint
    "checkpoint_dir_prefix": "checkpoints_convnext_unet",

    # Device / loader
    "device": "cuda",
    "num_workers": 4,
    "pin_memory": True,

    # Intensity augmentation
    "aug_use_brightness_contrast": True,
    "aug_use_gamma": True,
    "aug_use_gaussian_noise": True,
    "aug_brightness_range": 0.2,
    "aug_contrast_range": 0.2,
    "aug_gamma_range": [0.7, 1.3],
    "aug_noise_std": 0.05,

    # Geometry augmentation
    "aug_use_affine": True,
    "aug_affine_p": 0.8,
    "aug_affine_rotate_range": [-15, 15],
    "aug_affine_translate_frac": 0.1,
    "aug_affine_scale_range": [0.9, 1.1],
    "aug_affine_shear_range": [-5, 5],
    "aug_horizontal_flip_p": 0.5,
    "aug_random_crop_scale": [0.75, 1.0],
    "aug_random_crop_out_size": 512,
    "aug_random_crop_p": 0.5,
    "aug_elastic_alpha": 10,
    "aug_elastic_sigma": 6,
    "aug_elastic_p": 0.1,
    "aug_grid_distort_p": 0.05,
    "aug_gaussian_blur_p": 0.2,

    # Cutout
    "aug_cutout_p": 0.2,
    "aug_cutout_holes": 1,
    "aug_cutout_size_range": [0.05, 0.15],

    # Mean Teacher
    "use_mean_teacher": True,
    "mt_warmup_epochs": 20,
    "consistency_weight": 0.01,
    "ema_alpha": 0.996,
    "consistency_rampup_epochs": 80,
    "mt_ramp_epochs": 15,
    "consistency_confidence_threshold": 0.92,
    "consistency_cls_confidence_threshold": 0.90,
    "plaque_confidence_threshold": 0.92,
    "max_unlabeled_ratio": 1.0,

    # Mean Teacher strong augmentation
    "mean_teacher_strong": True,
    "mean_teacher_strong_brightness_range": 0.3,
    "mean_teacher_strong_contrast_range": 0.3,
    "mean_teacher_strong_gamma_range": [0.8, 1.2],
    "mean_teacher_strong_noise_std_range": [0.01, 0.04],
    "mean_teacher_strong_blur_p": 0.15,
    "mean_teacher_strong_cutout_p": 0.5,
    "mean_teacher_strong_cutout_holes": 3,
    "mean_teacher_strong_cutout_size_range": [0.05, 0.15],
    "unlabeled_use_geometry": False,

    # Plaque-guided segmentation consistency
    "use_plaque_confidence_mask": False,
    "use_plaque_topk_mask": True,
    "plaque_topk_ratio_start": 0.01,
    "plaque_topk_ratio_end": 0.03,
    "plaque_topk_ratio_ramp_epochs": 20,
    "plaque_topk_gamma": 2.0,

    # Consistency scheduling
    "seg_consistency_min_mask_ratio": 0.0,
    "seg_consistency_min_ratio_for_loss": 1e-4,
    "seg_consistency_weak_weight": 1.0,
    "use_cls_consistency": True,
    "cls_consistency_start_epoch": 30,
    "cls_consistency_weight": 0.015,
    "cls_consistency_min_ratio_for_loss": 1e-4,

    # Encoder freezing
    "encoder_freeze_epochs": 5,
    "encoder_freeze_stages": [0, 1, 2, 3],
    "encoder_freeze_stem": True,

    # Cross-validation
    "n_folds": 4,
    "fold": None,
    "start_fold": 0,
    "end_fold": 4,
}


# --------------------------------------------------------------------------
# 1. 4-Fold Split Creator
# --------------------------------------------------------------------------
def create_4fold_splits(split_data, dataset_path):
    def get_class_label(patient_id, dataset_path):
        label_path = os.path.join(dataset_path, "labels", f"{patient_id}_label.h5")
        try:
            with h5py.File(label_path, "r") as f:
                cls = f["cls"][()]
                if hasattr(cls, "__len__") and len(cls) > 0:
                    return int(cls[0])
                return int(cls)
        except Exception:
            return None

    def split_balanced(ids, n_groups):
        ids_per_group = len(ids) // n_groups
        groups = []
        for i in range(n_groups):
            start = i * ids_per_group
            end = start + ids_per_group if i < n_groups - 1 else len(ids)
            groups.append(ids[start:end])
        return groups

    all_patient_ids = set()
    for fold_data in split_data["splits"]:
        all_patient_ids.update(fold_data["train"])
        all_patient_ids.update(fold_data["val"])
    all_patient_ids = sorted(list(all_patient_ids))

    fold8_val = split_data["splits"][8]["val"]
    fold9_val = split_data["splits"][9]["val"]
    all_val_remaining = fold8_val + fold9_val

    val_id_to_class = {}
    for pid in all_val_remaining:
        cls = get_class_label(pid, dataset_path)
        if cls is not None:
            val_id_to_class[pid] = cls

    val_class_0_ids = [pid for pid, cls in val_id_to_class.items() if cls == 0]
    val_class_1_ids = [pid for pid, cls in val_id_to_class.items() if cls == 1]
    val_class_0_groups = split_balanced(val_class_0_ids, 4)
    val_class_1_groups = split_balanced(val_class_1_ids, 4)

    four_fold_splits = []
    fold_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]

    for fold_idx, (f1, f2) in enumerate(fold_pairs):
        val_ids = split_data["splits"][f1]["val"] + split_data["splits"][f2]["val"]
        val_ids.extend(val_class_0_groups[fold_idx])
        val_ids.extend(val_class_1_groups[fold_idx])

        val_set = set(val_ids)
        train_ids = [pid for pid in all_patient_ids if pid not in val_set]

        four_fold_splits.append({
            "fold": fold_idx,
            "train": train_ids,
            "val": val_ids,
        })

    return {
        "n_folds": 4,
        "total_patients": split_data.get("total_patients", 200),
        "class_distribution": split_data.get("class_distribution", {"0": 100, "1": 100}),
        "splits": four_fold_splits,
    }


# --------------------------------------------------------------------------
# 2. Utilities
# --------------------------------------------------------------------------
def _log_kfold_statistics(fold_results, group_checkpoint_dir):
    if len(fold_results) == 0:
        return

    best_scores = [r["best_score"] for r in fold_results]
    best_seg_scores = [r["best_seg_score"] for r in fold_results]
    best_cls_scores = [r["best_cls_score"] for r in fold_results]

    def compute_stats(scores):
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "max": np.max(scores),
            "min": np.min(scores),
        }

    total_stats = compute_stats(best_scores)
    seg_stats = compute_stats(best_seg_scores)
    cls_stats = compute_stats(best_cls_scores)

    print("\nK-FOLD SUMMARY")
    print(f"Total Best Scores: {total_stats['mean']:.4f} ± {total_stats['std']:.4f}")
    print(f"Segmentation Scores: {seg_stats['mean']:.4f} ± {seg_stats['std']:.4f}")
    print(f"Classification Scores: {cls_stats['mean']:.4f} ± {cls_stats['std']:.4f}")

    stats_file = os.path.join(group_checkpoint_dir, "statistics.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("K-FOLD CROSS-VALIDATION STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"\nTotal Best Scores (across {len(fold_results)} folds):\n")
        f.write(f"  Mean: {total_stats['mean']:.4f} ± {total_stats['std']:.4f}\n")
        f.write(f"\nSegmentation Scores:\n")
        f.write(f"  Mean: {seg_stats['mean']:.4f} ± {seg_stats['std']:.4f}\n")
        f.write(f"\nClassification Scores:\n")
        f.write(f"  Mean: {cls_stats['mean']:.4f} ± {cls_stats['std']:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nIndividual Fold Results:\n")
        for r in fold_results:
            f.write(
                f"Fold {r['fold']}: "
                f"Total={r['best_score']:.4f}, "
                f"Seg={r['best_seg_score']:.4f}, "
                f"Cls={r['best_cls_score']:.4f}\n"
            )

    print(f"Statistics saved to: {stats_file}")


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def process_mask_original(mask):
    new_mask = np.zeros_like(mask, dtype=np.int64)
    new_mask[mask == 128] = 1
    new_mask[mask == 255] = 2
    return new_mask


def load_original_masks(dataset_path, case_id):
    label_path = os.path.join(dataset_path, "labels", f"{case_id}_label.h5")
    with h5py.File(label_path, "r") as f:
        l_mask = f["long_mask"][:]
        t_mask = f["trans_mask"][:]
    l_mask = process_mask_original(l_mask)
    t_mask = process_mask_original(t_mask)
    return l_mask, t_mask


def _count_trainable_params(module: nn.Module) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _get_model_parts(model: nn.Module):
    encoder = getattr(model, "encoder", None)
    decoder = None
    if encoder is not None:
        decoder = [m for n, m in model.named_children() if n != "encoder"]
    return encoder, decoder


def _smooth_binary_targets(targets: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0:
        return targets
    return targets * (1.0 - eps) + 0.5 * eps


@torch.no_grad()
def _evaluate_model(model, val_loader, dataset_path, device):
    model.eval()

    dsc_p_long, nsd_p_long = AverageMeter(), AverageMeter()
    dsc_v_long, nsd_v_long = AverageMeter(), AverageMeter()
    dsc_p_trans, nsd_p_trans = AverageMeter(), AverageMeter()
    dsc_v_trans, nsd_v_trans = AverageMeter(), AverageMeter()

    cls_pred_list, cls_gt_list = [], []

    for l_img, t_img, l_mask, t_mask, cls, c_ids in val_loader:
        l_img, t_img = l_img.to(device), t_img.to(device)
        cls_gt = cls.to(device).float()

        if cls_gt.dim() == 0:
            cls_gt = cls_gt.unsqueeze(0).unsqueeze(0)
        elif cls_gt.dim() == 1:
            cls_gt = cls_gt.unsqueeze(1)

        segL, segT, cls_logits = model(l_img, t_img, return_cls=True)
        seg_logits = torch.cat([segL, segT], dim=0)

        cls_prob = torch.sigmoid(cls_logits)
        cls_pred = (cls_prob >= 0.5).long().view(-1)
        cls_pred_list.extend(cls_pred.cpu().numpy().tolist())
        cls_gt_list.extend(cls_gt.view(-1).cpu().numpy().tolist())

        B = l_img.shape[0]
        for b in range(B):
            case_id = c_ids[b]
            orig_l_mask, orig_t_mask = load_original_masks(dataset_path, case_id)

            l_logits_sample = seg_logits[b:b + 1]
            t_logits_sample = seg_logits[b + B:b + B + 1]

            h_orig_l, w_orig_l = orig_l_mask.shape
            l_pred_up = F.interpolate(
                l_logits_sample,
                size=(h_orig_l, w_orig_l),
                mode="bilinear",
                align_corners=False,
            )

            h_orig_t, w_orig_t = orig_t_mask.shape
            t_pred_up = F.interpolate(
                t_logits_sample,
                size=(h_orig_t, w_orig_t),
                mode="bilinear",
                align_corners=False,
            )

            l_pred_mask = torch.argmax(l_pred_up, dim=1).squeeze(0).cpu().numpy()
            t_pred_mask = torch.argmax(t_pred_up, dim=1).squeeze(0).cpu().numpy()

            dsc_p_long.update(dc(l_pred_mask == 1, orig_l_mask == 1))
            nsd_p_long.update(compute_nsd(l_pred_mask == 1, orig_l_mask == 1, tolerance=3.0))
            dsc_v_long.update(dc(l_pred_mask == 2, orig_l_mask == 2))
            nsd_v_long.update(compute_nsd(l_pred_mask == 2, orig_l_mask == 2, tolerance=3.0))

            dsc_p_trans.update(dc(t_pred_mask == 1, orig_t_mask == 1))
            nsd_p_trans.update(compute_nsd(t_pred_mask == 1, orig_t_mask == 1, tolerance=3.0))
            dsc_v_trans.update(dc(t_pred_mask == 2, orig_t_mask == 2))
            nsd_v_trans.update(compute_nsd(t_pred_mask == 2, orig_t_mask == 2, tolerance=3.0))

    S_long_vessel = (dsc_v_long.avg + nsd_v_long.avg) / 2
    S_long_plaque = (dsc_p_long.avg + nsd_p_long.avg) / 2
    S_trans_vessel = (dsc_v_trans.avg + nsd_v_trans.avg) / 2
    S_trans_plaque = (dsc_p_trans.avg + nsd_p_trans.avg) / 2

    cls_f1 = (
        f1_score(cls_gt_list, cls_pred_list, average="binary", pos_label=1, zero_division=0)
        if len(cls_gt_list) > 0 else 0.0
    )

    total_score = (
        cls_f1 * 0.4
        + S_long_vessel * 0.08
        + S_long_plaque * 0.12
        + S_trans_vessel * 0.08
        + S_trans_plaque * 0.12
    )
    final_seg_score = (
        S_long_vessel * 0.4
        + S_long_plaque * 0.6
        + S_trans_vessel * 0.4
        + S_trans_plaque * 0.6
    ) / 2

    return {
        "total_score": total_score,
        "final_seg_score": final_seg_score,
        "cls_f1": cls_f1,
        "metrics": {
            "long_vessel_dsc": dsc_v_long.avg,
            "long_vessel_nsd": nsd_v_long.avg,
            "long_plaque_dsc": dsc_p_long.avg,
            "long_plaque_nsd": nsd_p_long.avg,
            "trans_vessel_dsc": dsc_v_trans.avg,
            "trans_vessel_nsd": nsd_v_trans.avg,
            "trans_plaque_dsc": dsc_p_trans.avg,
            "trans_plaque_nsd": nsd_p_trans.avg,
        },
    }


# --------------------------------------------------------------------------
# 3. Main training loop
# --------------------------------------------------------------------------
def run_training(fold, cfg=None, group_checkpoint_dir=None):
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Fold: {fold}")

    device_str = cfg["device"]
    if device_str.lower() == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str.lower() == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    IMG_SIZE = cfg["img_size"]
    BATCH_SIZE = cfg["batch_size"]
    EPOCHS = cfg["train_epochs"]
    LR = cfg["learning_rate"]
    run_name = f"fold_{fold}"

    n_folds = cfg["n_folds"]
    if n_folds == 4:
        split_json_path = "split4.json"
        if not os.path.exists(split_json_path):
            dataset_path = cfg["data_root"]
            with open("split.json", "r") as f:
                original_splits = json.load(f)
            four_fold_splits = create_4fold_splits(original_splits, dataset_path)
            with open(split_json_path, "w") as f:
                json.dump(four_fold_splits, f, indent=2)
    else:
        split_json_path = cfg["split_json"]

    with open(split_json_path, "r") as f:
        split_data = json.load(f)

    train_ids = split_data["splits"][fold]["train"]
    val_ids = split_data["splits"][fold]["val"]
    dataset_path = cfg["data_root"]

    use_mean_teacher = cfg["use_mean_teacher"]
    unlabeled_ids = []
    if use_mean_teacher:
        images_dir = os.path.join(dataset_path, "images")
        if os.path.exists(images_dir):
            all_image_files = [f.replace(".h5", "") for f in os.listdir(images_dir) if f.endswith(".h5")]
            labeled_ids = set(train_ids + val_ids)
            unlabeled_ids = [fid for fid in all_image_files if fid not in labeled_ids]
        else:
            print("[Warning] images directory not found. Mean Teacher disabled.")
            use_mean_teacher = False

    aug_config = {
        "use_brightness_contrast": cfg["aug_use_brightness_contrast"],
        "use_gamma": cfg["aug_use_gamma"],
        "use_gaussian_noise": cfg["aug_use_gaussian_noise"],
        "brightness_range": cfg["aug_brightness_range"],
        "contrast_range": cfg["aug_contrast_range"],
        "gamma_range": cfg["aug_gamma_range"],
        "noise_std": cfg["aug_noise_std"],
    }
    aug_config.update({
        "use_affine": cfg["aug_use_affine"],
        "affine_rotate_range": cfg["aug_affine_rotate_range"],
        "affine_translate_frac": cfg["aug_affine_translate_frac"],
        "affine_scale_range": cfg["aug_affine_scale_range"],
        "affine_shear_range": cfg["aug_affine_shear_range"],
        "affine_p": cfg["aug_affine_p"],
        "horizontal_flip_p": cfg["aug_horizontal_flip_p"],
        "random_crop_scale": cfg["aug_random_crop_scale"],
        "random_crop_out_size": cfg["aug_random_crop_out_size"],
        "random_crop_p": cfg["aug_random_crop_p"],
        "elastic_alpha": cfg["aug_elastic_alpha"],
        "elastic_sigma": cfg["aug_elastic_sigma"],
        "elastic_p": cfg["aug_elastic_p"],
        "grid_distort_p": cfg["aug_grid_distort_p"],
        "gaussian_blur_p": cfg["aug_gaussian_blur_p"],
        "cutout_p": cfg["aug_cutout_p"],
        "cutout_holes": cfg["aug_cutout_holes"],
        "cutout_size_range": cfg["aug_cutout_size_range"],
        "unlabeled_use_geometry": cfg["unlabeled_use_geometry"],
    })

    mean_teacher_params = {
        "use_mean_teacher": use_mean_teacher,
        "consistency_weight": cfg["consistency_weight"],
        "ema_alpha": cfg["ema_alpha"],
        "consistency_rampup_epochs": cfg["consistency_rampup_epochs"],
        "consistency_confidence_threshold": cfg["consistency_confidence_threshold"],
        "consistency_cls_confidence_threshold": cfg["consistency_cls_confidence_threshold"],
        "mt_warmup_epochs": cfg["mt_warmup_epochs"],
        "max_unlabeled_ratio": cfg["max_unlabeled_ratio"],
        "use_plaque_confidence_mask": cfg["use_plaque_confidence_mask"],
        "plaque_confidence_threshold": cfg["plaque_confidence_threshold"],
        "mean_teacher_strong": cfg["mean_teacher_strong"],
        "mean_teacher_strong_brightness_range": cfg["mean_teacher_strong_brightness_range"],
        "mean_teacher_strong_contrast_range": cfg["mean_teacher_strong_contrast_range"],
        "mean_teacher_strong_gamma_range": cfg["mean_teacher_strong_gamma_range"],
        "mean_teacher_strong_noise_std_range": cfg["mean_teacher_strong_noise_std_range"],
        "mean_teacher_strong_blur_p": cfg["mean_teacher_strong_blur_p"],
        "mean_teacher_strong_cutout_p": cfg["mean_teacher_strong_cutout_p"],
        "mean_teacher_strong_cutout_holes": cfg["mean_teacher_strong_cutout_holes"],
        "mean_teacher_strong_cutout_size_range": cfg["mean_teacher_strong_cutout_size_range"],
    }
    aug_config["mean_teacher"] = mean_teacher_params

    train_dataset = CSVSemiDataset(dataset_path, train_ids, "train", size=IMG_SIZE, aug_config=aug_config)
    val_dataset = CSVSemiDataset(dataset_path, val_ids, "valid", size=IMG_SIZE, aug_config=None)

    unlabeled_dataset = None
    unlabeled_loader = None
    if use_mean_teacher and len(unlabeled_ids) > 0:
        unlabeled_dataset = CSVSemiDataset(dataset_path, unlabeled_ids, "unlabeled", size=IMG_SIZE, aug_config=aug_config)

    num_workers = cfg["num_workers"]
    pin_memory = cfg["pin_memory"]

    if use_mean_teacher and unlabeled_dataset is not None:
        labeled_batch_size = BATCH_SIZE // 2
        unlabeled_batch_size = BATCH_SIZE // 2

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=labeled_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=unlabeled_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if cfg["model_type"] != "convnext_unet":
        raise ValueError(f"Expected model_type='convnext_unet', got: {cfg['model_type']}")

    model = ConvNeXtUNet(
        in_chans=1,
        num_seg_classes=cfg["num_classes"],
        cls_class_num=cfg["cls_class_num"],
        pretrained_encoder=cfg["pretrained_encoder"],
        convnext_model=cfg["convnext_model"],
        cls_hidden=cfg["cls_hidden"],
        cls_dropout=cfg["cls_dropout"],
        plaque_gate_alpha=cfg["plaque_gate_alpha"],
        morph_proj_dim=cfg["morph_proj_dim"],
    ).to(device)

    print(f"Model: ConvNeXtUNet, backbone={cfg['convnext_model']}")

    encoder_freeze_epochs = cfg["encoder_freeze_epochs"]
    if encoder_freeze_epochs and hasattr(model, "freeze_encoder"):
        model.freeze_encoder(
            stages=cfg["encoder_freeze_stages"],
            freeze_stem=cfg["encoder_freeze_stem"],
        )
        print(
            f"Encoder frozen for first {encoder_freeze_epochs} epochs "
            f"(stages={cfg['encoder_freeze_stages']}, stem={cfg['encoder_freeze_stem']})"
        )

    teacher_model = None
    if use_mean_teacher:
        teacher_model = ConvNeXtUNet(
            in_chans=1,
            num_seg_classes=cfg["num_classes"],
            cls_class_num=cfg["cls_class_num"],
            pretrained_encoder=cfg["pretrained_encoder"],
            convnext_model=cfg["convnext_model"],
            cls_hidden=cfg["cls_hidden"],
            cls_dropout=cfg["cls_dropout"],
            plaque_gate_alpha=cfg["plaque_gate_alpha"],
            morph_proj_dim=cfg["morph_proj_dim"],
        ).to(device)
        teacher_model.load_state_dict(model.state_dict())
        for p in teacher_model.parameters():
            p.requires_grad = False
        print("Mean Teacher enabled: teacher initialized from student")

    dice_loss_fn = DiceLoss(n_classes=cfg["num_classes"])
    focal_loss_fn = smp.losses.FocalLoss(mode="multiclass")
    cls_loss_fn = nn.BCEWithLogitsLoss()

    param_groups = []
    added_param_ids = set()

    def _add_params(module, lr):
        if module is None:
            return
        params = [p for p in module.parameters() if p.requires_grad and id(p) not in added_param_ids]
        if not params:
            return
        for p in params:
            added_param_ids.add(id(p))
        param_groups.append({"params": params, "lr": lr})

    _add_params(getattr(model, "encoder", None), cfg["encoder_lr"])
    _add_params(getattr(model, "decoder_long", None), cfg["decoder_lr"])
    _add_params(getattr(model, "decoder_trans", None), cfg["decoder_lr"])
    _add_params(getattr(model, "cls_head", None), cfg["cls_head_lr"])

    remaining_params = [p for p in model.parameters() if p.requires_grad and id(p) not in added_param_ids]
    if remaining_params:
        param_groups.append({"params": remaining_params, "lr": LR})

    optimizer = optim.AdamW(param_groups, lr=LR, weight_decay=cfg["weight_decay"])
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

    accumulation_steps = cfg["gradient_accumulation_steps"]
    cls_warmup_epochs = cfg["cls_warmup_epochs"]
    cls_loss_weight = cfg["cls_loss_weight"]
    cls_label_smoothing = float(cfg["cls_label_smoothing"])

    consistency_weight = cfg["consistency_weight"] if use_mean_teacher else 0.0
    ema_alpha = cfg["ema_alpha"]
    mt_warmup_epochs = cfg["mt_warmup_epochs"]
    mt_ramp_epochs = cfg["mt_ramp_epochs"]
    max_unlabeled_ratio = cfg["max_unlabeled_ratio"]

    use_plaque_confidence_mask = cfg["use_plaque_confidence_mask"]
    plaque_confidence_threshold = cfg["plaque_confidence_threshold"]

    use_plaque_topk_mask = cfg["use_plaque_topk_mask"]
    plaque_topk_ratio_start = cfg["plaque_topk_ratio_start"]
    plaque_topk_ratio_end = cfg["plaque_topk_ratio_end"]
    plaque_topk_ratio_ramp_epochs = cfg["plaque_topk_ratio_ramp_epochs"]
    plaque_topk_gamma = cfg["plaque_topk_gamma"]

    seg_consistency_min_mask_ratio = cfg["seg_consistency_min_mask_ratio"]
    seg_consistency_min_ratio_for_loss = cfg["seg_consistency_min_ratio_for_loss"]
    seg_consistency_weak_weight = cfg["seg_consistency_weak_weight"]

    use_cls_consistency = cfg["use_cls_consistency"]
    cls_consistency_start_epoch = cfg["cls_consistency_start_epoch"]
    cls_consistency_weight = cfg["cls_consistency_weight"]
    cls_consistency_min_ratio_for_loss = cfg["cls_consistency_min_ratio_for_loss"]

    best_score_st = 0.0
    best_score_tea = 0.0
    seg_score_at_best_st = 0.0
    cls_score_at_best_st = 0.0
    seg_score_at_best_tea = 0.0
    cls_score_at_best_tea = 0.0

    early_stopping = EarlyStopping(
        patience=cfg["early_stopping_patience"],
        verbose=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_root = cfg["checkpoint_dir_prefix"]

    if group_checkpoint_dir is not None:
        checkpoint_dir = os.path.join(group_checkpoint_dir, run_name)
        base_checkpoint_dir = group_checkpoint_dir
    else:
        base_checkpoint_dir = checkpoint_root
        os.makedirs(base_checkpoint_dir, exist_ok=True)
        checkpoint_dir = os.path.join(base_checkpoint_dir, f"{run_name}_{timestamp}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file = os.path.join(checkpoint_dir, "log.txt")

    class Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    log_file_handle = open(log_file, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file_handle)

    global_step = 0

    for epoch in range(EPOCHS):
        if encoder_freeze_epochs and epoch == encoder_freeze_epochs and hasattr(model, "unfreeze_encoder"):
            model.unfreeze_encoder()
            print(f"Encoder unfrozen at epoch {epoch}")

        if use_mean_teacher and teacher_model is not None and epoch == mt_warmup_epochs:
            teacher_model.load_state_dict(model.state_dict())
            print(f"[MeanTeacher] Teacher reset from student at epoch {epoch}")

        scale = (1 - epoch / max(1, EPOCHS)) ** 0.9
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * scale

        print(f"\n===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}, Previous best: {best_score_st:.4f}")

        encoder_module, decoder_modules = _get_model_parts(model)
        encoder_trainable = _count_trainable_params(encoder_module)
        decoder_trainable = 0
        if decoder_modules:
            for m in decoder_modules:
                decoder_trainable += _count_trainable_params(m)
        print(f"Trainable params - encoder: {encoder_trainable}, decoder/others: {decoder_trainable}")

        model.train()
        if teacher_model is not None:
            teacher_model.eval()

        optimizer.zero_grad(set_to_none=True)

        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_cons_loss_weighted = 0.0
        train_cons_seg = 0.0
        train_cons_seg_scaled = 0.0
        train_cons_cls = 0.0
        train_mask_ratio = 0.0
        train_mask_ratio_cls = 0.0
        train_mask_mean = 0.0
        num_batches = 0

        use_mt_epoch = use_mean_teacher and unlabeled_loader is not None and epoch >= mt_warmup_epochs

        if use_mt_epoch:
            max_unlabeled_steps = int(len(train_loader) * max_unlabeled_ratio)
            num_batches_epoch = min(len(unlabeled_loader), max_unlabeled_steps) if max_unlabeled_steps > 0 else len(unlabeled_loader)
        else:
            num_batches_epoch = len(train_loader)

        if use_mt_epoch:
            train_loader_iter = iter(train_loader)

            for i, (ul_img_strong, ut_img_strong, ul_img_weak, ut_img_weak, _) in enumerate(
                tqdm(unlabeled_loader, desc=f"Epoch {epoch}")
            ):
                if max_unlabeled_steps > 0 and i >= max_unlabeled_steps:
                    break

                try:
                    l_img, t_img, l_mask, t_mask, cls, _ = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    l_img, t_img, l_mask, t_mask, cls, _ = next(train_loader_iter)

                l_img, t_img = l_img.to(device), t_img.to(device)
                l_mask, t_mask = l_mask.to(device), t_mask.to(device)
                cls = cls.to(device).float()

                if cls.dim() == 0:
                    cls = cls.unsqueeze(0).unsqueeze(0)
                elif cls.dim() == 1:
                    cls = cls.unsqueeze(1)

                segL_l, segT_l, cls_logits_l = model(l_img, t_img, return_cls=True)

                seg_logits_l = torch.cat([segL_l, segT_l], dim=0)
                l_masks = torch.cat([l_mask, t_mask], dim=0)

                loss_dice = dice_loss_fn(seg_logits_l, l_masks, softmax=True)
                loss_focal = focal_loss_fn(seg_logits_l, l_masks)
                seg_loss = loss_focal + loss_dice

                if epoch < cls_warmup_epochs:
                    cls_loss = torch.tensor(0.0, device=device)
                else:
                    cls_targets = _smooth_binary_targets(cls, cls_label_smoothing)
                    cls_loss = cls_loss_fn(cls_logits_l, cls_targets)

                loss_sup = seg_loss + cls_loss * cls_loss_weight

                ul_weak, ut_weak = ul_img_weak.to(device), ut_img_weak.to(device)
                ul_strong, ut_strong = ul_img_strong.to(device), ut_img_strong.to(device)

                with torch.no_grad():
                    segL_u_tea, segT_u_tea, cls_logits_u_teacher = teacher_model(ul_weak, ut_weak, return_cls=True)
                    seg_logits_u_teacher = torch.cat([segL_u_tea, segT_u_tea], dim=0)

                    probs_teacher_seg = torch.softmax(seg_logits_u_teacher, dim=1)
                    plaque_probs = probs_teacher_seg[:, 1, ...]

                    if use_plaque_topk_mask:
                        if plaque_topk_ratio_ramp_epochs > 0:
                            progress = (epoch - mt_warmup_epochs) / float(plaque_topk_ratio_ramp_epochs)
                            progress = max(0.0, min(1.0, progress))
                            k_ratio = plaque_topk_ratio_start + (plaque_topk_ratio_end - plaque_topk_ratio_start) * progress
                        else:
                            k_ratio = plaque_topk_ratio_end

                        k_ratio = float(np.clip(k_ratio, 1e-6, 1.0))
                        flat = plaque_probs.view(plaque_probs.shape[0], -1)
                        num_pixels = flat.shape[1]
                        k = max(1, int(k_ratio * num_pixels))
                        topk_vals, _ = torch.topk(flat, k, dim=1, largest=True, sorted=True)
                        thresh = topk_vals[:, -1].unsqueeze(1)
                        mask_select = (flat >= thresh).view_as(plaque_probs)

                        weights = torch.clamp(plaque_probs, 0.0, 1.0)
                        if plaque_topk_gamma != 1.0:
                            weights = weights ** plaque_topk_gamma

                        mask = mask_select.float() * weights
                        mask_ratio = mask_select.float().mean().item()
                        mask_mean = mask.mean().item()

                    elif use_plaque_confidence_mask:
                        t = plaque_confidence_threshold
                        mask = torch.clamp((plaque_probs - t) / (1.0 - t + 1e-8), 0.0, 1.0)
                        mask_ratio = (mask > 0).float().mean().item()
                        mask_mean = mask.mean().item()

                    else:
                        max_probs, _ = torch.max(probs_teacher_seg, dim=1)
                        seg_conf_thresh = cfg["consistency_confidence_threshold"]
                        mask = (max_probs >= seg_conf_thresh).float()
                        mask_ratio = (mask > 0).float().mean().item()
                        mask_mean = mask.mean().item()

                    prob_cls_teacher = torch.sigmoid(cls_logits_u_teacher)
                    if prob_cls_teacher.dim() == 1:
                        prob_cls_teacher = prob_cls_teacher.unsqueeze(1)

                    conf_cls = torch.max(prob_cls_teacher, 1 - prob_cls_teacher)
                    cls_conf_thresh = cfg["consistency_cls_confidence_threshold"]
                    mask_cls = (conf_cls >= cls_conf_thresh).float()
                    mask_ratio_cls = mask_cls.mean().item()

                segL_u, segT_u, cls_logits_u = model(ul_strong, ut_strong, return_cls=True)
                seg_logits_u = torch.cat([segL_u, segT_u], dim=0)

                probs_student_seg = torch.softmax(seg_logits_u, dim=1)
                if probs_student_seg.shape[-2:] != probs_teacher_seg.shape[-2:]:
                    probs_student_seg = F.interpolate(
                        probs_student_seg,
                        size=probs_teacher_seg.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                if (mask_ratio < seg_consistency_min_ratio_for_loss) or (mask.sum().item() < 1.0):
                    loss_cons_seg = torch.tensor(0.0, device=device)
                else:
                    per_pixel = ((probs_student_seg - probs_teacher_seg) ** 2).mean(dim=1)
                    loss_cons_seg = (per_pixel * mask).sum() / (mask.sum() + 1e-8)

                prob_cls_student = torch.sigmoid(cls_logits_u)
                if prob_cls_student.dim() == 1:
                    prob_cls_student = prob_cls_student.unsqueeze(1)

                seg_cons_scale = 1.0 if mask_ratio >= seg_consistency_min_mask_ratio else seg_consistency_weak_weight

                if (
                    (not use_cls_consistency)
                    or (epoch < cls_consistency_start_epoch)
                    or (mask_ratio_cls < cls_consistency_min_ratio_for_loss)
                ):
                    loss_cons_cls = torch.tensor(0.0, device=device)
                else:
                    loss_cons_cls = (((prob_cls_student - prob_cls_teacher) ** 2) * mask_cls).sum() / (mask_cls.sum() + 1e-8)

                loss_cons_seg_scaled = seg_cons_scale * loss_cons_seg

                mt_progress = max(0, epoch - mt_warmup_epochs + 1)
                ramp = min(1.0, mt_progress / max(1, mt_ramp_epochs))
                cur_cons_weight = consistency_weight * ramp
                cur_cls_cons_weight = cls_consistency_weight * ramp

                loss = (
                    loss_sup
                    + cur_cons_weight * loss_cons_seg_scaled
                    + cur_cls_cons_weight * loss_cons_cls
                ) / accumulation_steps

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                update_ema_variables(model, teacher_model, alpha=ema_alpha, global_step=global_step)
                global_step += 1

                train_loss += loss.item() * accumulation_steps
                train_seg_loss += seg_loss.item()
                train_cls_loss += cls_loss.item()
                train_cons_loss_weighted += (
                    cur_cons_weight * loss_cons_seg_scaled + cur_cls_cons_weight * loss_cons_cls
                ).item()
                train_cons_seg += loss_cons_seg.item()
                train_cons_seg_scaled += loss_cons_seg_scaled.item()
                train_cons_cls += loss_cons_cls.item()
                train_mask_ratio += mask_ratio
                train_mask_mean += mask_mean
                train_mask_ratio_cls += mask_ratio_cls

                num_batches += 1

        else:
            for i, (l_img, t_img, l_mask, t_mask, cls, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                l_img, t_img = l_img.to(device), t_img.to(device)
                l_mask, t_mask = l_mask.to(device), t_mask.to(device)
                cls = cls.to(device).float()

                if cls.dim() == 0:
                    cls = cls.unsqueeze(0).unsqueeze(0)
                elif cls.dim() == 1:
                    cls = cls.unsqueeze(1)

                segL, segT, cls_logits = model(l_img, t_img, return_cls=True)

                seg_logits = torch.cat([segL, segT], dim=0)
                masks = torch.cat([l_mask, t_mask], dim=0)

                loss_dice = dice_loss_fn(seg_logits, masks, softmax=True)
                loss_focal = focal_loss_fn(seg_logits, masks)
                seg_loss = loss_focal + loss_dice

                if epoch < cls_warmup_epochs:
                    cls_loss = torch.tensor(0.0, device=device)
                else:
                    cls_targets = _smooth_binary_targets(cls, cls_label_smoothing)
                    cls_loss = cls_loss_fn(cls_logits, cls_targets)

                loss = (seg_loss + cls_loss * cls_loss_weight) / accumulation_steps
                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += loss.item() * accumulation_steps
                train_seg_loss += seg_loss.item()
                train_cls_loss += cls_loss.item()

                num_batches += 1

        if num_batches > 0 and (num_batches % accumulation_steps != 0):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_train_loss = train_loss / max(1, num_batches)
        avg_seg = train_seg_loss / max(1, num_batches)
        avg_cls = train_cls_loss / max(1, num_batches)

        if use_mt_epoch:
            print(
                f"Loss_total: {avg_train_loss:.4f} | "
                f"Seg: {avg_seg:.4f} | "
                f"Cls: {avg_cls:.4f} | "
                f"Cons(seg): {train_cons_seg / max(1, num_batches):.4f} | "
                f"Cons(seg_w): {train_cons_seg_scaled / max(1, num_batches):.4f} | "
                f"Cons(cls): {train_cons_cls / max(1, num_batches):.4f} | "
                f"Cons(w): {train_cons_loss_weighted / max(1, num_batches):.4f} | "
                f"Mask(seg): {train_mask_ratio / max(1, num_batches):.4f} | "
                f"Mask(cls): {train_mask_ratio_cls / max(1, num_batches):.4f}"
            )
        else:
            print(f"Loss_total: {avg_train_loss:.4f} | Seg: {avg_seg:.4f} | Cls: {avg_cls:.4f}")

        st_eval = _evaluate_model(model, val_loader, dataset_path, device)
        total_score = st_eval["total_score"]
        final_seg_score = st_eval["final_seg_score"]
        cls_f1 = st_eval["cls_f1"]

        print(
            f"[Student] Val total: {total_score:.4f} | "
            f"Seg: {final_seg_score:.4f} | "
            f"Cls: {cls_f1:.4f}"
        )

        tea_eval = None
        if use_mean_teacher and teacher_model is not None:
            tea_eval = _evaluate_model(teacher_model, val_loader, dataset_path, device)
            print(
                f"[Teacher] Val total: {tea_eval['total_score']:.4f} | "
                f"Seg: {tea_eval['final_seg_score']:.4f} | "
                f"Cls: {tea_eval['cls_f1']:.4f}"
            )

        if total_score > best_score_st:
            best_score_st = total_score
            seg_score_at_best_st = final_seg_score
            cls_score_at_best_st = cls_f1
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_st_fold{fold}_model.pth"))
            print(f"[Student] New Best Score: {best_score_st:.4f}")

        if tea_eval is not None and tea_eval["total_score"] > best_score_tea:
            best_score_tea = tea_eval["total_score"]
            seg_score_at_best_tea = tea_eval["final_seg_score"]
            cls_score_at_best_tea = tea_eval["cls_f1"]
            torch.save(teacher_model.state_dict(), os.path.join(checkpoint_dir, f"best_tea_fold{fold}_model.pth"))
            print(f"[Teacher] New Best Score: {best_score_tea:.4f}")

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"latest_fold{fold}_model.pth"))

        early_stopping(total_score)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    sys.stdout = original_stdout
    log_file_handle.close()

    if group_checkpoint_dir is None:
        final_score_str = f"{best_score_st:.4f}".replace(".", "_")
        new_dir = os.path.join(base_checkpoint_dir, f"{run_name}_{timestamp}_score_{final_score_str}")
        os.rename(checkpoint_dir, new_dir)

    return {
        "fold": fold,
        "best_score": best_score_st,
        "best_seg_score": seg_score_at_best_st,
        "best_cls_score": cls_score_at_best_st,
    }


# --------------------------------------------------------------------------
# 4. Entry
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--start_fold", type=int, default=None)
    parser.add_argument("--end_fold", type=int, default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()

    if args.fold is not None:
        run_training(args.fold, cfg=cfg)
    else:
        config_fold = cfg["fold"]
        if config_fold is not None:
            run_training(config_fold, cfg=cfg)
        else:
            n_folds = cfg["n_folds"]
            start = args.start_fold if args.start_fold is not None else cfg["start_fold"]
            end = args.end_fold if args.end_fold is not None else cfg["end_fold"]

            base_cp = cfg["checkpoint_dir_prefix"]
            grp_name = "kfold_summary"
            grp_cp = os.path.join(base_cp, grp_name)
            os.makedirs(grp_cp, exist_ok=True)

            results = []
            for f in range(start, end):
                res = run_training(f, cfg=cfg, group_checkpoint_dir=grp_cp)
                if res is not None:
                    results.append(res)
                torch.cuda.empty_cache()

            _log_kfold_statistics(results, grp_cp)