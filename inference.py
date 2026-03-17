import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F


"""
Inference script for CSV2026 submission.

Expected input:
    DATA_ROOT/images/*.h5
Each input HDF5 file must contain:
    - long_img
    - trans_img

Output:
    OUTPUT_DIR/{case_name}_pred.h5
Each output HDF5 file contains:
    - long_mask
    - trans_mask
    - cls
"""


class ValH5Dataset:
    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.paths = sorted(self.images_dir.glob("*.h5"))
        if not self.paths:
            raise RuntimeError(f"No .h5 files found in {images_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        with h5py.File(path, "r") as f:
            long_img = f["long_img"][:]
            trans_img = f["trans_img"][:]

        long_shape = long_img.shape
        trans_shape = trans_img.shape

        long_img = torch.from_numpy(long_img).unsqueeze(0).float()
        trans_img = torch.from_numpy(trans_img).unsqueeze(0).float()

        if long_img.max() > 1.0:
            long_img = long_img / 255.0
        if trans_img.max() > 1.0:
            trans_img = trans_img / 255.0

        return str(path), long_img, trans_img, long_shape, trans_shape


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for CSV2026 submission")
    parser.add_argument("--data-root", type=str, default=None, help="Path to dataset root containing images/")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save predictions")
    parser.add_argument("--resize-target", type=int, default=None, help="Input resize target")
    parser.add_argument("--gpu", type=str, default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--cls-threshold", type=float, default=None, help="Classification threshold")
    return parser.parse_args()


def load_runtime_config(args):
    data_root = Path(args.data_root or os.getenv("DATA_ROOT", "/example_data"))
    model_path = Path(args.model_path or os.getenv("MODEL_PATH", "/weights/best.pth"))
    output_dir = Path(args.output_dir or os.getenv("OUTPUT_DIR", "/output/preds"))
    resize_target = int(args.resize_target or os.getenv("RESIZE_TARGET", "512"))
    gpu = str(args.gpu or os.getenv("GPU", "0"))
    cls_threshold = float(args.cls_threshold or os.getenv("CLS_THRESHOLD", "0.8"))
    return data_root, model_path, output_dir, resize_target, gpu, cls_threshold


def get_device(gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_mask_for_submission(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8)


def run_inference(model, dataset, output_dir: Path, resize_target: int, cls_threshold: float, device):
    print(f"[INFO] Running inference on {len(dataset)} files")
    print(f"[INFO] Saving predictions to: {output_dir}")
    print(f"[INFO] Classification threshold: {cls_threshold}")

    for idx in range(len(dataset)):
        path, long_img, trans_img, long_shape, trans_shape = dataset[idx]

        x_long = long_img.unsqueeze(0).to(device)
        x_trans = trans_img.unsqueeze(0).to(device)

        x_long_resized = F.interpolate(
            x_long,
            size=(resize_target, resize_target),
            mode="bilinear",
            align_corners=False,
        )
        x_trans_resized = F.interpolate(
            x_trans,
            size=(resize_target, resize_target),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            seg_long_logits, seg_trans_logits, cls_logits = model(x_long_resized, x_trans_resized)

            seg_long_up = F.interpolate(
                seg_long_logits,
                size=long_shape,
                mode="bilinear",
                align_corners=False,
            )
            seg_trans_up = F.interpolate(
                seg_trans_logits,
                size=trans_shape,
                mode="bilinear",
                align_corners=False,
            )

            pred_long = torch.argmax(seg_long_up, dim=1).squeeze(0).cpu().numpy()
            pred_trans = torch.argmax(seg_trans_up, dim=1).squeeze(0).cpu().numpy()

            cls_prob = torch.sigmoid(cls_logits).cpu().numpy()
            cls_pred = (cls_prob >= cls_threshold).astype(np.uint8).reshape(-1)

        pred_long = encode_mask_for_submission(pred_long)
        pred_trans = encode_mask_for_submission(pred_trans)

        case_name = Path(path).stem
        out_path = output_dir / f"{case_name}_pred.h5"

        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("long_mask", data=pred_long, compression="gzip")
            hf.create_dataset("trans_mask", data=pred_trans, compression="gzip")
            hf.create_dataset("cls", data=cls_pred)

        if idx % 10 == 0 or idx == len(dataset) - 1:
            print(f"[INFO] [{idx + 1}/{len(dataset)}] Saved: {out_path}")

    print(f"[INFO] Inference done. Total cases = {len(dataset)}")


def main():
    args = parse_args()
    data_root, model_path, output_dir, resize_target, gpu, cls_threshold = load_runtime_config(args)
    device = get_device(gpu)

    print(f"[INFO] DATA_ROOT={data_root}")
    print(f"[INFO] MODEL_PATH={model_path}")
    print(f"[INFO] OUTPUT_DIR={output_dir}")
    print(f"[INFO] RESIZE_TARGET={resize_target}")
    print(f"[INFO] GPU={gpu}")
    print(f"[INFO] DEVICE={device}")

    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {model_path}")

    images_dir = data_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    from models.Model import build_submission_model, load_weights_compat

    model = build_submission_model().to(device)
    model.eval()
    print("[INFO] Model built.")

    model = load_weights_compat(model, str(model_path), map_location=device)
    print("[INFO] Model weights loaded.")

    dataset = ValH5Dataset(images_dir)
    run_inference(model, dataset, output_dir, resize_target, cls_threshold, device)


if __name__ == "__main__":
    main()