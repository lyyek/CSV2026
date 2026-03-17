import os
import random

import h5py
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from dataset.transform import (
    random_brightness_contrast,
    random_gamma,
    random_gaussian_noise,
    random_affine,
    random_crop_resize,
    elastic_transform,
    grid_distortion,
    coarse_dropout,
    gaussian_blur,
    mean_teacher_strong_intensity,
)


class CSVSemiDataset(Dataset):
    def __init__(self, data_root, case_ids, mode, size=224, aug_config=None):
        self.data_root = data_root
        self.case_ids = case_ids
        self.mode = mode
        self.size = size

        self.aug_config = aug_config or {}

        # intensity augmentation
        self.use_brightness_contrast = self.aug_config.get("use_brightness_contrast", False)
        self.use_gamma = self.aug_config.get("use_gamma", False)
        self.use_gaussian_noise = self.aug_config.get("use_gaussian_noise", False)

        self.brightness_range = self.aug_config.get("brightness_range", 0.2)
        self.contrast_range = self.aug_config.get("contrast_range", 0.2)
        self.gamma_range = tuple(self.aug_config.get("gamma_range", [0.7, 1.3]))
        self.noise_std = self.aug_config.get("noise_std", 0.05)

        # geometry augmentation
        self.use_affine = self.aug_config.get("use_affine", True)
        self.affine_rotate_range = tuple(self.aug_config.get("affine_rotate_range", [-15, 15]))
        self.affine_translate_frac = float(self.aug_config.get("affine_translate_frac", 0.1))
        self.affine_scale_range = tuple(self.aug_config.get("affine_scale_range", [0.9, 1.1]))
        self.affine_shear_range = tuple(self.aug_config.get("affine_shear_range", [-5, 5]))
        self.affine_p = float(self.aug_config.get("affine_p", 0.8))

        self.horizontal_flip_p = float(self.aug_config.get("horizontal_flip_p", 0.0))

        self.random_crop_scale = tuple(self.aug_config.get("random_crop_scale", [0.8, 1.0]))
        self.random_crop_out_size = int(self.aug_config.get("random_crop_out_size", self.size))
        self.random_crop_p = float(self.aug_config.get("random_crop_p", 0.5))

        self.elastic_alpha = float(self.aug_config.get("elastic_alpha", 10))
        self.elastic_sigma = float(self.aug_config.get("elastic_sigma", 6))
        self.elastic_p = float(self.aug_config.get("elastic_p", 0.2))

        self.grid_distort_p = float(self.aug_config.get("grid_distort_p", 0.15))
        self.gaussian_blur_p = float(self.aug_config.get("gaussian_blur_p", 0.2))

        self.cutout_p = float(self.aug_config.get("cutout_p", 0.2))
        self.cutout_holes = int(self.aug_config.get("cutout_holes", 1))
        self.cutout_size_range = tuple(self.aug_config.get("cutout_size_range", [0.05, 0.15]))

        # unlabeled augmentation option
        self.unlabeled_use_geometry = bool(self.aug_config.get("unlabeled_use_geometry", True))

    def __len__(self):
        return len(self.case_ids)

    def _maybe_horizontal_flip(self, img, mask=None, p=0.0):
        if p <= 0 or random.random() > p:
            return img, mask

        img = np.flip(img, axis=1).copy()
        if mask is not None:
            mask = np.flip(mask, axis=1).copy()
        return img, mask

    def _process_mask(self, mask):
        new_mask = np.zeros_like(mask, dtype=np.int64)
        new_mask[mask == 128] = 1   # plaque
        new_mask[mask == 255] = 2   # vessel
        return new_mask

    def _read_pair(self, image_h5_file):
        with h5py.File(image_h5_file, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]

        long_img = long_img.astype(np.float32)
        trans_img = trans_img.astype(np.float32)

        try:
            if long_img.max() > 1.0:
                long_img = long_img / 255.0
        except ValueError:
            pass

        try:
            if trans_img.max() > 1.0:
                trans_img = trans_img / 255.0
        except ValueError:
            pass

        return long_img, trans_img

    def _read_label(self, label_h5_file):
        with h5py.File(label_h5_file, "r") as f:
            long_mask = self._process_mask(f["long_mask"][:])
            trans_mask = self._process_mask(f["trans_mask"][:])
            cls = f["cls"][()]
        return long_mask, trans_mask, cls

    def _resize_pair_and_masks(self, long_img, trans_img, long_mask=None, trans_mask=None):
        x, y = long_img.shape
        long_img = zoom(long_img, (self.size / x, self.size / y), order=1)

        x2, y2 = trans_img.shape
        trans_img = zoom(trans_img, (self.size / x2, self.size / y2), order=1)

        if long_mask is not None:
            long_mask = zoom(long_mask, (self.size / x, self.size / y), order=0)

        if trans_mask is not None:
            trans_mask = zoom(trans_mask, (self.size / x2, self.size / y2), order=0)

        return long_img, trans_img, long_mask, trans_mask

    def __getitem__(self, item):
        case_id = self.case_ids[item]
        img_path = os.path.join(self.data_root, "images", f"{case_id}.h5")

        if self.mode in ["train", "train_l", "valid"]:
            label_path = os.path.join(self.data_root, "labels", f"{case_id}_label.h5")

            long_img, trans_img = self._read_pair(img_path)
            long_mask, trans_mask, cls = self._read_label(label_path)

            if self.mode != "valid":
                # shared geometric augmentation
                long_img, long_mask = self._maybe_horizontal_flip(long_img, long_mask, p=self.horizontal_flip_p)
                trans_img, trans_mask = self._maybe_horizontal_flip(trans_img, trans_mask, p=self.horizontal_flip_p)

                if self.use_affine and self.affine_p > 0:
                    long_img, long_mask = random_affine(
                        long_img,
                        long_mask,
                        rotate_range=self.affine_rotate_range,
                        translate_frac=self.affine_translate_frac,
                        scale_range=self.affine_scale_range,
                        shear_range=self.affine_shear_range,
                        p=self.affine_p,
                    )
                    trans_img, trans_mask = random_affine(
                        trans_img,
                        trans_mask,
                        rotate_range=self.affine_rotate_range,
                        translate_frac=self.affine_translate_frac,
                        scale_range=self.affine_scale_range,
                        shear_range=self.affine_shear_range,
                        p=self.affine_p,
                    )

                if self.random_crop_p > 0:
                    long_img, long_mask = random_crop_resize(
                        long_img,
                        long_mask,
                        scale_range=self.random_crop_scale,
                        out_size=self.random_crop_out_size,
                        p=self.random_crop_p,
                    )
                    trans_img, trans_mask = random_crop_resize(
                        trans_img,
                        trans_mask,
                        scale_range=self.random_crop_scale,
                        out_size=self.random_crop_out_size,
                        p=self.random_crop_p,
                    )

                if self.elastic_p > 0:
                    long_img, long_mask = elastic_transform(
                        long_img, long_mask,
                        alpha=self.elastic_alpha,
                        sigma=self.elastic_sigma,
                        p=self.elastic_p,
                    )
                    trans_img, trans_mask = elastic_transform(
                        trans_img, trans_mask,
                        alpha=self.elastic_alpha,
                        sigma=self.elastic_sigma,
                        p=self.elastic_p,
                    )

                if self.grid_distort_p > 0:
                    long_img, long_mask = grid_distortion(long_img, long_mask, p=self.grid_distort_p)
                    trans_img, trans_mask = grid_distortion(trans_img, trans_mask, p=self.grid_distort_p)

                # intensity augmentation
                if self.use_brightness_contrast:
                    long_img, _ = random_brightness_contrast(
                        long_img, None,
                        self.brightness_range,
                        self.contrast_range,
                    )
                    trans_img, _ = random_brightness_contrast(
                        trans_img, None,
                        self.brightness_range,
                        self.contrast_range,
                    )

                if self.use_gamma:
                    long_img, _ = random_gamma(long_img, None, self.gamma_range)
                    trans_img, _ = random_gamma(trans_img, None, self.gamma_range)

                if self.use_gaussian_noise:
                    long_img, _ = random_gaussian_noise(long_img, None, self.noise_std)
                    trans_img, _ = random_gaussian_noise(trans_img, None, self.noise_std)

                if self.gaussian_blur_p > 0:
                    long_img, _ = gaussian_blur(long_img, None, p=self.gaussian_blur_p)
                    trans_img, _ = gaussian_blur(trans_img, None, p=self.gaussian_blur_p)

                if self.cutout_p > 0:
                    long_img, _ = coarse_dropout(
                        long_img, None,
                        holes=self.cutout_holes,
                        size_range=self.cutout_size_range,
                        p=self.cutout_p,
                    )
                    trans_img, _ = coarse_dropout(
                        trans_img, None,
                        holes=self.cutout_holes,
                        size_range=self.cutout_size_range,
                        p=self.cutout_p,
                    )

            long_img, trans_img, long_mask, trans_mask = self._resize_pair_and_masks(
                long_img, trans_img, long_mask, trans_mask
            )

            return (
                torch.from_numpy(long_img).unsqueeze(0).float(),
                torch.from_numpy(trans_img).unsqueeze(0).float(),
                torch.from_numpy(long_mask).long(),
                torch.from_numpy(trans_mask).long(),
                torch.tensor(cls).long(),
                case_id,
            )

        elif self.mode == "unlabeled":
            long_img, trans_img = self._read_pair(img_path)

            if self.unlabeled_use_geometry:
                long_img, _ = self._maybe_horizontal_flip(long_img, None, p=self.horizontal_flip_p)
                trans_img, _ = self._maybe_horizontal_flip(trans_img, None, p=self.horizontal_flip_p)

                if self.use_affine and self.affine_p > 0:
                    long_img, _ = random_affine(
                        long_img,
                        None,
                        rotate_range=self.affine_rotate_range,
                        translate_frac=self.affine_translate_frac,
                        scale_range=self.affine_scale_range,
                        shear_range=self.affine_shear_range,
                        p=self.affine_p,
                    )
                    trans_img, _ = random_affine(
                        trans_img,
                        None,
                        rotate_range=self.affine_rotate_range,
                        translate_frac=self.affine_translate_frac,
                        scale_range=self.affine_scale_range,
                        shear_range=self.affine_shear_range,
                        p=self.affine_p,
                    )

                if self.random_crop_p > 0:
                    long_img, _ = random_crop_resize(
                        long_img,
                        None,
                        scale_range=self.random_crop_scale,
                        out_size=self.random_crop_out_size,
                        p=self.random_crop_p,
                    )
                    trans_img, _ = random_crop_resize(
                        trans_img,
                        None,
                        scale_range=self.random_crop_scale,
                        out_size=self.random_crop_out_size,
                        p=self.random_crop_p,
                    )

                if self.elastic_p > 0:
                    long_img, _ = elastic_transform(
                        long_img, None,
                        alpha=self.elastic_alpha,
                        sigma=self.elastic_sigma,
                        p=self.elastic_p,
                    )
                    trans_img, _ = elastic_transform(
                        trans_img, None,
                        alpha=self.elastic_alpha,
                        sigma=self.elastic_sigma,
                        p=self.elastic_p,
                    )

                if self.grid_distort_p > 0:
                    long_img, _ = grid_distortion(long_img, None, p=self.grid_distort_p)
                    trans_img, _ = grid_distortion(trans_img, None, p=self.grid_distort_p)

            long_img_weak = long_img.copy()
            trans_img_weak = trans_img.copy()

            mt = self.aug_config.get("mean_teacher", {}) if isinstance(self.aug_config, dict) else {}

            long_img, _ = mean_teacher_strong_intensity(
                long_img,
                None,
                brightness_range=mt.get("mean_teacher_strong_brightness_range", 0.3),
                contrast_range=mt.get("mean_teacher_strong_contrast_range", 0.3),
                gamma_range=tuple(mt.get("mean_teacher_strong_gamma_range", [0.6, 1.6])),
                noise_std_range=tuple(mt.get("mean_teacher_strong_noise_std_range", [0.02, 0.08])),
                blur_p=mt.get("mean_teacher_strong_blur_p", 0.35),
                cutout_p=mt.get("mean_teacher_strong_cutout_p", 0.9),
                cutout_holes=mt.get("mean_teacher_strong_cutout_holes", 3),
                cutout_size_range=tuple(mt.get("mean_teacher_strong_cutout_size_range", [0.05, 0.15])),
            )
            trans_img, _ = mean_teacher_strong_intensity(
                trans_img,
                None,
                brightness_range=mt.get("mean_teacher_strong_brightness_range", 0.3),
                contrast_range=mt.get("mean_teacher_strong_contrast_range", 0.3),
                gamma_range=tuple(mt.get("mean_teacher_strong_gamma_range", [0.6, 1.6])),
                noise_std_range=tuple(mt.get("mean_teacher_strong_noise_std_range", [0.02, 0.08])),
                blur_p=mt.get("mean_teacher_strong_blur_p", 0.35),
                cutout_p=mt.get("mean_teacher_strong_cutout_p", 0.9),
                cutout_holes=mt.get("mean_teacher_strong_cutout_holes", 3),
                cutout_size_range=tuple(mt.get("mean_teacher_strong_cutout_size_range", [0.05, 0.15])),
            )

            x, y = long_img.shape
            long_img = zoom(long_img, (self.size / x, self.size / y), order=1)
            long_img_weak = zoom(long_img_weak, (self.size / x, self.size / y), order=1)

            x2, y2 = trans_img.shape
            trans_img = zoom(trans_img, (self.size / x2, self.size / y2), order=1)
            trans_img_weak = zoom(trans_img_weak, (self.size / x2, self.size / y2), order=1)

            return (
                torch.from_numpy(long_img).unsqueeze(0).float(),
                torch.from_numpy(trans_img).unsqueeze(0).float(),
                torch.from_numpy(long_img_weak).unsqueeze(0).float(),
                torch.from_numpy(trans_img_weak).unsqueeze(0).float(),
                case_id,
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")