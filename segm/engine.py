#FULLY SUPERVISED

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
from segm.metrics import gather_data
import segm.utils.torch as ptu

IGNORE_LABEL = 255

# ---------------------------------------------------------
# LOSS TRACKING
# ---------------------------------------------------------
LOSS_HISTORY = {
    "CE": [],
    "Weighted_CE": [],
    "Dice": [],
    "Validation": [],
    "Total": [],
}

# ---------------------------------------------------------
# HELPER: REMAP MASK LABELS
# ---------------------------------------------------------
def remap_mask(mask, label_map=None):
    """
    Remap mask values to 0..n_cls-1
    label_map: dict mapping original labels to [0..n_cls-1]
    """
    if label_map is None:
        label_map = {0: 0, 1: 1, 2: 2, 3: 3}  # update if needed
    remapped = np.full_like(mask, fill_value=IGNORE_LABEL, dtype=np.int64)
    for k, v in label_map.items():
        remapped[mask == k] = v
    return remapped

# ---------------------------------------------------------
# LOSS FUNCTIONS
# ---------------------------------------------------------
def dice_loss(pred, target, smooth=1e-6):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# ---------------------------------------------------------
# PLOT LOSSES
# ---------------------------------------------------------
def plot_losses(log_dir):
    plt.figure(figsize=(10, 6))
    max_len = max(len(v) for v in LOSS_HISTORY.values())
    x_axis = np.arange(1, max_len + 1)

    style = {
        "CE": ("blue", "o"),
        "Weighted_CE": ("orange", "s"),
        "Dice": ("green", "D"),
        "Validation": ("red", "x"),
        "Total": ("purple", "^"),
    }

    for key in ["CE", "Weighted_CE", "Dice", "Validation", "Total"]:
        values = LOSS_HISTORY.get(key, [])
        if len(values) > 0:
            plot_values = values + [np.nan] * (max_len - len(values))
            color, marker = style.get(key, ("black", "o"))
            plt.plot(
                x_axis,
                plot_values,
                label=key.replace("_", " "),
                linewidth=2.0,
                color=color,
                marker=marker,
                markersize=6,
                alpha=0.85
            )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Losses", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_losses.png"), dpi=150)
    plt.close()

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_one_epoch(model, data_loader, optimizer, lr_scheduler, epoch, amp_autocast,
                    loss_scaler=None, log_dir=None, class_weights=None, val_loader=None):

    model.train()
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    if class_weights is not None:
        weighted_ce_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(ptu.device),
            ignore_index=IGNORE_LABEL
        )
    else:
        weighted_ce_fn = ce_loss_fn

    n_cls = getattr(data_loader.dataset, "n_cls", 4)

    ce_epoch, weighted_ce_epoch, dice_epoch, total_epoch = 0.0, 0.0, 0.0, 0.0

    for batch_idx, batch in enumerate(data_loader):
        images = batch["image"].to(ptu.device)
        masks = batch["mask"].to(ptu.device).long()

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(images)

            if batch_idx == 0:
                print("DEBUG: Model output shape:", outputs.shape)
                if outputs.shape[1] != n_cls:
                    raise ValueError(
                        f"Model output channels ({outputs.shape[1]}) != dataset classes ({n_cls})"
                    )

            ce_loss = ce_loss_fn(outputs, masks)
            weighted_ce_loss = weighted_ce_fn(outputs, masks)

            probs = torch.softmax(outputs, dim=1)
            if probs.shape[1] > 1:
                dice = torch.mean(torch.stack([
                    dice_loss(probs[:, c, :, :], (masks == c).float())
                    for c in range(probs.shape[1])
                ]))
            else:
                dice = dice_loss(probs[:, 0, :, :], masks.float())

            total_loss = ce_loss + dice

        if loss_scaler is not None:
            loss_scaler(total_loss, optimizer)
        else:
            total_loss.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        ce_epoch += ce_loss.item()
        weighted_ce_epoch += weighted_ce_loss.item()
        dice_epoch += dice.item()
        total_epoch += total_loss.item()

    n_batches = len(data_loader)
    ce_epoch /= n_batches
    weighted_ce_epoch /= n_batches
    dice_epoch /= n_batches
    total_epoch /= n_batches

    val_loss_epoch = None
    if val_loader is not None:
        val_loss_epoch = compute_validation_loss(
            model, val_loader, ce_loss_fn, weighted_ce_fn, amp_autocast
        )

    LOSS_HISTORY["CE"].append(ce_epoch)
    LOSS_HISTORY["Weighted_CE"].append(weighted_ce_epoch)
    LOSS_HISTORY["Dice"].append(dice_epoch)
    LOSS_HISTORY["Validation"].append(val_loss_epoch if val_loss_epoch is not None else np.nan)
    LOSS_HISTORY["Total"].append(total_epoch)

    if log_dir:
        plot_losses(log_dir)

    return {
        "CE": ce_epoch,
        "Weighted_CE": weighted_ce_epoch,
        "Dice": dice_epoch,
        "Validation": val_loss_epoch,
        "Total": total_epoch,
    }

# ---------------------------------------------------------
# VALIDATION LOSS
# ---------------------------------------------------------
def compute_validation_loss(model, val_loader, ce_fn, weighted_ce_fn, amp_autocast):
    model.eval()
    ce_sum, weighted_ce_sum, dice_sum, total_sum = 0.0, 0.0, 0.0, 0.0
    n_batches = len(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(ptu.device)
            masks = batch["mask"].to(ptu.device).long()

            with amp_autocast():
                outputs = model(images)

                ce_loss = ce_fn(outputs, masks)
                weighted_ce_loss = weighted_ce_fn(outputs, masks)

                probs = torch.softmax(outputs, dim=1)
                if probs.shape[1] > 1:
                    dice = torch.mean(torch.stack([
                        dice_loss(probs[:, c, :, :], (masks == c).float())
                        for c in range(probs.shape[1])
                    ]))
                else:
                    dice = dice_loss(probs[:, 0, :, :], masks.float())

                total_loss = ce_loss + dice

            ce_sum += ce_loss.item()
            weighted_ce_sum += weighted_ce_loss.item()
            dice_sum += dice.item()
            total_sum += total_loss.item()

    return total_sum / n_batches

# ---------------------------------------------------------
# EVALUATION WITH FULL DATASET DIAGNOSTICS
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(model, data_loader, val_seg_gt_raw, window_size=None, window_stride=None,
             amp_autocast=None, log_dir=None, epoch=None):

    model_eval = model.module if hasattr(model, "module") else model
    seg_pred = {}

    n_cls = getattr(data_loader.dataset, "n_cls", 4)
    gt_pixel_count = np.zeros(n_cls, dtype=np.int64)
    pred_pixel_count = np.zeros(n_cls, dtype=np.int64)
    class_image_count = np.zeros(n_cls, dtype=np.int64)
    total_images = 0

    dataset_label_summary = {c: [] for c in range(n_cls)}  # track image IDs per class

    for batch in data_loader:
        images = batch["image"].to(ptu.device)
        ids = batch["id"]

        with amp_autocast():
            outputs = model_eval(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        for i, file_id in enumerate(ids):
            pred = preds[i]
            gt_raw = val_seg_gt_raw[file_id]
            gt_raw = remap_mask(gt_raw)

            unique_labels = np.unique(gt_raw)
            for lbl in unique_labels:
                if lbl != IGNORE_LABEL:
                    dataset_label_summary[lbl].append(file_id)

            if pred.shape != gt_raw.shape:
                import cv2
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (gt_raw.shape[1], gt_raw.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            seg_pred[file_id] = pred

            gt_flat = gt_raw.flatten()
            pred_flat = pred.flatten()

            for c in range(n_cls):
                gt_pixel_count[c] += np.sum(gt_flat == c)
                pred_pixel_count[c] += np.sum(pred_flat == c)
                if np.any(gt_flat == c):
                    class_image_count[c] += 1

            total_images += 1

    # Per-class image occurrence debug
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - True Image Occurrence per Class")
    print("-"*70)
    print("Total images evaluated:", total_images)
    for c in range(n_cls):
        print(f"Class {c} appears in {class_image_count[c]} images (sample: {dataset_label_summary[c][:5]})")
    print("="*70 + "\n")

    # Per-class pixel statistics debug
    print("\n" + "="*70)
    print(f"[DEBUG] Epoch {epoch} - Per-Class Pixel Statistics")
    print("-"*70)
    for c in range(n_cls):
        print("Class {}: GT pixels = {:<12} | Predicted pixels = {}".format(
            c, gt_pixel_count[c], pred_pixel_count[c]
        ))
    print("="*70 + "\n")

    # Compute metrics
    seg_pred = gather_data(seg_pred)
    val_seg_gt_filtered = {
        k: np.asarray(remap_mask(val_seg_gt_raw[k]), dtype=np.int64)
        for k in seg_pred.keys()
    }

    metrics = compute_segmentation_metrics(seg_pred, val_seg_gt_filtered, n_cls)

    # CSV logging
    if log_dir and epoch is not None:
        csv_path = os.path.join(log_dir, "evaluation_metrics.csv")
        header = ["epoch"] + list(metrics.keys())
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            row = {"epoch": epoch}
            for k, v in metrics.items():
                row[k] = list(v) if isinstance(v, np.ndarray) else v
            writer.writerow(row)

    return metrics

# ---------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------
def compute_segmentation_metrics(preds, gts, n_cls):
    eps = 1e-6

    total_pixels = 0
    correct_pixels = 0

    dice_class = np.zeros(n_cls, dtype=float)
    precision_class = np.zeros(n_cls, dtype=float)
    recall_class = np.zeros(n_cls, dtype=float)
    iou_class = np.zeros(n_cls, dtype=float)
    acc_class = np.zeros(n_cls, dtype=float)

    # For FWIoU
    fw_intersection = np.zeros(n_cls, dtype=float)
    fw_gt_pixels = np.zeros(n_cls, dtype=float)

    # For global IoU
    total_intersection = 0
    total_union = 0

    for k in preds.keys():
        pred = preds[k].flatten()
        gt = gts[k].flatten()

        mask_valid = (gt != IGNORE_LABEL)
        pred = pred[mask_valid]
        gt = gt[mask_valid]

        total_pixels += len(gt)
        correct_pixels += np.sum(pred == gt)

        for c in range(n_cls):
            pred_c = (pred == c)
            gt_c = (gt == c)

            intersection = np.sum(pred_c & gt_c)
            union = np.sum(pred_c | gt_c)

            dice_class[c] += (2 * intersection) / (np.sum(pred_c) + np.sum(gt_c) + eps)
            precision_class[c] += intersection / (np.sum(pred_c) + eps)
            recall_class[c] += intersection / (np.sum(gt_c) + eps)
            iou_class[c] += intersection / (union + eps)
            acc_class[c] += np.sum(pred_c & gt_c) / (np.sum(gt_c) + eps)

            # FWIoU stats per class
            fw_intersection[c] += intersection
            fw_gt_pixels[c] += np.sum(gt_c)

            # Global IoU
            total_intersection += intersection
            total_union += union

    num_images = len(preds)

    dice_class /= num_images
    precision_class /= num_images
    recall_class /= num_images
    iou_class /= num_images
    acc_class /= num_images

    # ============================
    # Compute final metrics
    # ============================
    pixel_acc = correct_pixels / (total_pixels + eps)
    mean_acc = np.mean(acc_class)
    mean_iou = np.mean(iou_class)

    # Proper FWIoU:
    fw_iou = (fw_intersection / (fw_gt_pixels + eps)).sum()

    # Global IoU over all pixels
    global_iou = total_intersection / (total_union + eps)

    metrics = {
        "PixelAcc": pixel_acc,
        "MeanAcc": mean_acc,
        "MeanIoU": mean_iou,    # per-class average
        "FWIoU": fw_iou,        # frequency-weighted IoU
        "IoU": global_iou,      # global pixel-wise IoU
        "PerClassIoU": iou_class,
        "PerClassDice": dice_class,
        "Precision": precision_class,
        "Recall": recall_class,
        "F1": 2 * (precision_class * recall_class) /
              (precision_class + recall_class + eps),
    }

    return metrics


