# -*- coding: utf-8 -*-
import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import pandas as pd
from tabulate import tabulate

import segm.utils.torch as ptu
from segm.data.utils import STATS, dataset_cat_description
from segm.model.factory import load_model
from segm.model.utils import inference

# Dataset imports
from segm.data.ade20k import ADE20KSegmentation
from segm.data.flame import FlameDataset
from segm.data.meta import MetaDataset

IGNORE_LABEL = 255

# ---------------------
# Helper: Convert segmentation map to RGB
# ---------------------
def seg_to_rgb(seg, colors):
    """
    Convert segmentation map to RGB image using the provided colors.
    Args:
        seg: numpy array or torch tensor of shape [H,W] or [1,H,W]
        colors: numpy array of shape [n_classes, 3], dtype=np.uint8
    Returns:
        RGB image as numpy array of shape [H,W,3], dtype=np.uint8
    """
    if torch.is_tensor(seg):
        seg = seg.cpu().numpy()
    if seg.ndim == 3 and seg.shape[0] == 1:
        seg = seg[0]

    h, w = seg.shape
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(seg):
        if cls_id >= len(colors):
            continue
        seg_rgb[seg == cls_id] = colors[cls_id]
    return seg_rgb

# ---------------------
# Main inference function
# ---------------------
@click.command()
@click.option("--model-path", type=str, required=True, help="Path to trained model checkpoint (.pth)")
@click.option("--input-dir", "-i", type=str, required=True, help="Folder with input images")
@click.option("--output-dir", "-o", type=str, required=True, help="Folder where output images will be saved")
@click.option("--gpu/--cpu", default=True, is_flag=True, help="Use GPU if available")
@click.option("--gt-dir", type=str, default=None, help="Folder with ground truth masks (optional)")
def main(model_path, input_dir, output_dir, gpu, gt_dir):
    """Run inference for semantic segmentation and optionally compute evaluation metrics."""
    ptu.set_gpu_mode(gpu)

    print("\n?? Loading model...")
    model, variant = load_model(model_path)
    model.to(ptu.device)
    model.eval()

    # ---- Dataset selection ----
    dataset_name = variant.get("dataset") or variant.get("dataset_kwargs", {}).get("dataset", "ade20k")
    normalization_key = variant.get("dataset_kwargs", {}).get("normalization", "default")

    if dataset_name.lower() == "ade20k":
        from segm.data.ade20k import ADE20K_CATS_PATH as CATS_PATH
        DatasetClass = ADE20KSegmentation
    elif dataset_name.lower() == "flame":
        from segm.data.flame import FLAME_CATS_PATH as CATS_PATH
        DatasetClass = FlameDataset
    elif dataset_name.lower() == "meta":
        from segm.data.meta import META_CATS_PATH as CATS_PATH
        DatasetClass = MetaDataset
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    cat_names, cat_colors = dataset_cat_description(CATS_PATH)
    print("\n[DEBUG] Loaded categories and colors:")
    for i, c in enumerate(cat_colors):
        print(f"  Class {i}: {c}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    list_dir = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    print(f"Found {len(list_dir)} images in {input_dir}\n")

    preds, filenames = [], []

    # ---- Inference params ----
    image_size = variant.get("inference_kwargs", {}).get("window_size", 512)
    window_stride = variant.get("inference_kwargs", {}).get("window_stride", image_size)
    batch_size = variant.get("inference_kwargs", {}).get("batch_size", 2)

    gt_dir = Path(gt_dir) if gt_dir else None

    # ---- Vibrant colors for prediction & GT overlay ----
    vibrant_colors = np.array([
        [0, 0, 0],       # background
        [255, 0, 0],     # class 1 ? red
        [0, 255, 0],     # class 2 ? green
        [0, 0, 255],     # class 3 ? blue
    ], dtype=np.uint8)

    # ---- Inference loop ----
    for filename in tqdm(list_dir, ncols=80, desc="Running inference"):
        pil_im = Image.open(filename).convert("RGB")
        im = F.pil_to_tensor(pil_im).float() / 255.0
        im = F.normalize(im, STATS[normalization_key]["mean"], STATS[normalization_key]["std"])
        im = im.to(ptu.device).unsqueeze(0)

        logits = inference(
            model,
            [im],
            [{"flip": False}],
            ori_shape=im.shape[2:4],
            window_size=image_size,
            window_stride=window_stride,
            batch_size=batch_size,
        )

        # ---- Convert logits to segmentation map ----
        seg_map = logits.argmax(0)  # [H, W]
        seg_map_np = seg_map.cpu().numpy() if torch.is_tensor(seg_map) else seg_map

        # ---- Convert prediction to RGB ----
        seg_rgb = seg_to_rgb(seg_map_np, vibrant_colors)
        pil_seg = Image.fromarray(seg_rgb)

        # ---- Overlay prediction on original image ----
        pil_pred_overlay = Image.blend(pil_im, pil_seg, 0.5)

        # ---- Handle GT overlay if provided ----
        if gt_dir is not None:
            gt_path = gt_dir / (filename.stem + ".png")
            if gt_path.exists():
                gt_mask_np = np.array(Image.open(gt_path))
                print("\n[DEBUG] GT file:", gt_path.name, "Unique values:", np.unique(gt_mask_np))

                gt_color = np.zeros((gt_mask_np.shape[0], gt_mask_np.shape[1], 3), dtype=np.uint8)
                for cls_id in np.unique(gt_mask_np):
                    if cls_id == IGNORE_LABEL:
                        gt_color[gt_mask_np == IGNORE_LABEL] = (50, 50, 50)
                        continue
                    if cls_id < 0 or cls_id >= len(vibrant_colors):
                        print(f"[WARNING] GT value {cls_id} has no color entry")
                        continue
                    gt_color[gt_mask_np == cls_id] = vibrant_colors[cls_id]

                pil_gt_color = Image.fromarray(gt_color)
                if pil_gt_color.size != pil_pred_overlay.size:
                    pil_gt_color = pil_gt_color.resize(pil_pred_overlay.size, Image.NEAREST)

                # Combine GT and prediction overlays side by side
                combined = Image.new("RGB", (pil_gt_color.width + pil_pred_overlay.width, pil_gt_color.height))
                combined.paste(pil_gt_color, (0, 0))
                combined.paste(pil_pred_overlay, (pil_gt_color.width, 0))
                combined.save(output_dir / filename.name)
            else:
                pil_pred_overlay.save(output_dir / filename.name)
        else:
            pil_pred_overlay.save(output_dir / filename.name)

        preds.append(seg_map_np)  # ensure numpy
        filenames.append(filename.name)

    print(f"\n? Inference complete. Results saved to: {output_dir}")

    # ---- Evaluation ----
    if gt_dir is not None:
        print("\n?? Running evaluation...")

        # Accept multiple split names
        possible_splits = ["val", "validation"]
        split_name = None
        for split in possible_splits:
            if (gt_dir.parents[1] / "images" / split).exists() and (gt_dir.parents[1] / "masks" / split).exists():
                split_name = split
                break
        if split_name is None:
            raise RuntimeError(f"No valid split folder found in {gt_dir.parents[1]} (checked {possible_splits})")

        val_dataset = DatasetClass(
            image_size=image_size,
            crop_size=image_size,
            split=split_name,
            normalization=normalization_key,
            root=gt_dir.parents[1]
        )
        print(f"Using dataset split: {split_name}")

        n_cls = val_dataset.n_cls
        gt_seg_maps = {p.stem: np.array(Image.open(p)) for p in sorted(gt_dir.glob("*.png"))}

        intersection = np.zeros(n_cls)
        union = np.zeros(n_cls)
        gt_count = np.zeros(n_cls)
        pred_count = np.zeros(n_cls)
        tp = np.zeros(n_cls)
        fp = np.zeros(n_cls)
        fn = np.zeros(n_cls)
        total_pixels, correct_pixels = 0, 0

        for fname, pred in zip(filenames, preds):
            key = Path(fname).stem
            if key not in gt_seg_maps:
                continue
            gt = gt_seg_maps[key]  # numpy array
            pred = pred if isinstance(pred, np.ndarray) else np.array(pred)  # ensure numpy
            mask = gt != IGNORE_LABEL
            total_pixels += mask.sum()
            correct_pixels += (pred[mask] == gt[mask]).sum()

            for cls in range(n_cls):
                pred_cls = pred == cls
                gt_cls = gt == cls
                inter = np.logical_and(pred_cls, gt_cls).sum()
                uni = np.logical_or(pred_cls, gt_cls).sum()
                intersection[cls] += inter
                union[cls] += uni
                gt_count[cls] += gt_cls.sum()
                pred_count[cls] += pred_cls.sum()
                tp[cls] += inter
                fp[cls] += np.logical_and(pred_cls, np.logical_not(gt_cls)).sum()
                fn[cls] += np.logical_and(np.logical_not(pred_cls), gt_cls).sum()

        # Compute metrics
        pixel_acc = correct_pixels / (total_pixels + 1e-10)
        per_class_acc = intersection / (gt_count + 1e-10)
        mean_acc = per_class_acc.mean()
        per_class_iou = intersection / (union + 1e-10)
        mean_iou = np.nanmean(per_class_iou)
        fw_iou = np.sum(gt_count * per_class_iou) / np.sum(gt_count)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2*precision*recall/(precision+recall+1e-10)
        dice = 2*intersection/(gt_count + pred_count + 1e-10)

        # Save metrics to CSV
        metrics_dict = {
            "Pixel_Acc": pixel_acc,
            "Mean_Acc": mean_acc,
            "Mean_IoU": mean_iou,
            "FWIoU": fw_iou,
            "Dice": np.mean(dice),
            "PerClassDice": dice.tolist(),
            "Precision": np.mean(precision),
            "Recall": np.mean(recall),
            "F1": np.mean(f1),
        }

        csv_path = output_dir / "eval_metrics.csv"
        pd.DataFrame([metrics_dict]).to_csv(csv_path, index=False)
        print(f"\n?? Metrics saved to: {csv_path}")

        # Print per-class table
        table = []
        for i in range(n_cls):
            table.append([
                i,
                cat_names[i] if i < len(cat_names) else f"class_{i}",
                round(per_class_acc[i],4),
                round(per_class_iou[i],4),
                round(dice[i],4),
                round(precision[i],4),
                round(recall[i],4),
                round(f1[i],4)
            ])
        print("\nPer-class metrics:")
        print(tabulate(table, headers=["ID","Name","Acc","IoU","Dice","Precision","Recall","F1"]))

if __name__ == "__main__":
    main()
