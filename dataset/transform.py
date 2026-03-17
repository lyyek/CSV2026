import math
import random

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import map_coordinates, gaussian_filter


def random_brightness_contrast(img, mask=None, brightness_range=0.2, contrast_range=0.2, p=0.5):
    if random.random() > p:
        return img, mask

    brightness = np.random.uniform(1 - brightness_range, 1 + brightness_range)
    img = img * brightness

    contrast = np.random.uniform(1 - contrast_range, 1 + contrast_range)
    img = (img - 0.5) * contrast + 0.5

    img = np.clip(img, 0.0, 1.0)
    return img, mask


def random_gamma(img, mask=None, gamma_range=(0.7, 1.3), p=0.5):
    if random.random() > p:
        return img, mask

    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    img = np.power(img, gamma)
    img = np.clip(img, 0.0, 1.0)
    return img, mask


def random_gaussian_noise(img, mask=None, noise_std=0.05, p=0.5):
    if random.random() > p:
        return img, mask

    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img, mask


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img

    arr = np.array(img)

    if arr.dtype == np.uint8:
        out = arr
    else:
        if np.issubdtype(arr.dtype, np.floating) or (arr.max() <= 1.0 and arr.min() >= 0.0):
            out = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            out = arr.astype(np.uint8)

    if out.ndim == 2:
        return Image.fromarray(out, mode="L")
    if out.shape[2] == 3:
        return Image.fromarray(out)
    return Image.fromarray(out[:, :, 0])


def _to_numpy(img):
    arr = np.array(img)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    return arr


def random_affine(
    img,
    mask=None,
    rotate_range=(-15, 15),
    translate_frac=0.1,
    scale_range=(0.9, 1.1),
    shear_range=(-5, 5),
    p=0.5,
):
    if random.random() > p:
        return img, mask

    h, w = img.shape[:2]
    angle = np.random.uniform(rotate_range[0], rotate_range[1])
    tx = np.random.uniform(-translate_frac, translate_frac) * w
    ty = np.random.uniform(-translate_frac, translate_frac) * h
    scale = np.random.uniform(scale_range[0], scale_range[1])
    shear = math.radians(np.random.uniform(shear_range[0], shear_range[1]))

    center = (w * 0.5, h * 0.5)

    angle_rad = math.radians(angle)
    a = scale * math.cos(angle_rad)
    b = scale * math.sin(angle_rad + shear)
    d = -scale * math.sin(angle_rad)
    e = scale * math.cos(angle_rad + shear)

    c = center[0] - a * center[0] - b * center[1] + tx
    f = center[1] - d * center[0] - e * center[1] + ty

    pil_img = _to_pil(img)
    pil_img = pil_img.transform(
        (w, h),
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
    )

    if mask is not None:
        pil_mask = _to_pil(mask)
        pil_mask = pil_mask.transform(
            (w, h),
            Image.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.NEAREST,
        )
        return _to_numpy(pil_img), np.array(pil_mask)

    return _to_numpy(pil_img), mask


def random_crop_resize(img, mask=None, scale_range=(0.8, 1.0), out_size=512, p=0.5):
    if random.random() > p:
        return img, mask

    h, w = img.shape[:2]
    scale = np.random.uniform(scale_range[0], scale_range[1])
    crop_h = int(h * scale)
    crop_w = int(w * scale)

    if crop_h == h and crop_w == w:
        if out_size != h:
            pil_img = _to_pil(img).resize((out_size, out_size), resample=Image.BILINEAR)
            if mask is not None:
                pil_mask = _to_pil(mask).resize((out_size, out_size), resample=Image.NEAREST)
                return _to_numpy(pil_img), np.array(pil_mask)
            return _to_numpy(pil_img), mask
        return img, mask

    x = np.random.randint(0, w - crop_w + 1)
    y = np.random.randint(0, h - crop_h + 1)

    pil_img = _to_pil(img).crop((x, y, x + crop_w, y + crop_h)).resize(
        (out_size, out_size), resample=Image.BILINEAR
    )

    if mask is not None:
        pil_mask = _to_pil(mask).crop((x, y, x + crop_w, y + crop_h)).resize(
            (out_size, out_size), resample=Image.NEAREST
        )
        return _to_numpy(pil_img), np.array(pil_mask)

    return _to_numpy(pil_img), mask


def elastic_transform(image, mask=None, alpha=10, sigma=6, p=0.5):
    if random.random() > p:
        return image, mask

    shape = image.shape[:2]
    random_state = np.random.RandomState(None)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channel = map_coordinates(image[:, :, c], indices, order=1, mode="reflect").reshape(shape)
            channels.append(channel)
        img_warp = np.stack(channels, axis=2)
    else:
        img_warp = map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)

    if mask is not None:
        mask_warp = map_coordinates(mask, indices, order=0, mode="reflect").reshape(shape)
        return np.clip(img_warp, 0.0, 1.0), mask_warp

    return np.clip(img_warp, 0.0, 1.0), mask


def grid_distortion(image, mask=None, distort_limit=0.03, num_steps=5, p=0.2):
    if random.random() > p:
        return image, mask

    h, w = image.shape[:2]
    alpha = max(h, w) * distort_limit
    sigma = max(2, int(max(h, w) * 0.01))
    return elastic_transform(image, mask=mask, alpha=alpha, sigma=sigma, p=1.0)


def coarse_dropout(img, mask=None, holes=1, size_range=(0.05, 0.15), p=0.5, fill_value=0.0):
    if random.random() > p:
        return img, mask

    h, w = img.shape[:2]
    for _ in range(np.random.randint(1, holes + 1)):
        frac = np.random.uniform(size_range[0], size_range[1])
        hole_h = int(h * frac)
        hole_w = int(w * frac)
        x = np.random.randint(0, max(1, w - hole_w + 1))
        y = np.random.randint(0, max(1, h - hole_h + 1))

        if img.ndim == 3:
            img[y:y + hole_h, x:x + hole_w, :] = fill_value
        else:
            img[y:y + hole_h, x:x + hole_w] = fill_value

        if mask is not None:
            mask[y:y + hole_h, x:x + hole_w] = 0

    return img, mask


def gaussian_blur(img, mask=None, p=0.2, sigma_range=(0.1, 1.5)):
    if random.random() > p:
        return img, mask

    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    pil_img = _to_pil(img).filter(ImageFilter.GaussianBlur(radius=sigma))
    return _to_numpy(pil_img), mask


def mean_teacher_strong_intensity(
    img,
    mask=None,
    *,
    brightness_range=0.3,
    contrast_range=0.3,
    gamma_range=(0.6, 1.6),
    noise_std_range=(0.02, 0.08),
    blur_p=0.45,
    blur_sigma_range=(0.2, 1.8),
    cutout_p=0.9,
    cutout_holes=3,
    cutout_size_range=(0.05, 0.15),
    brightness_contrast_p=0.95,
    gamma_p=0.95,
    noise_p=0.75,
):
    img, mask = random_brightness_contrast(
        img,
        mask,
        brightness_range=brightness_range,
        contrast_range=contrast_range,
        p=brightness_contrast_p,
    )
    img, mask = random_gamma(img, mask, gamma_range=gamma_range, p=gamma_p)

    if noise_std_range is None:
        noise_std = np.random.uniform(0.02, 0.08)
    else:
        noise_std = np.random.uniform(noise_std_range[0], noise_std_range[1])

    img, mask = random_gaussian_noise(img, mask, noise_std=noise_std, p=noise_p)
    img, mask = gaussian_blur(img, mask, p=blur_p, sigma_range=blur_sigma_range)
    img, mask = coarse_dropout(
        img,
        mask,
        holes=cutout_holes,
        size_range=cutout_size_range,
        p=cutout_p,
        fill_value=0.0,
    )
    return img, mask