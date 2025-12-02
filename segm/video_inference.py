import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

from segm.model.factory import load_model
from segm.data import utils

IGNORE_LABEL = 255

def load_colormap(meta_yml_path):
    with open(meta_yml_path, "r") as f:
        cats = yaml.load(f, Loader=yaml.FullLoader)
    # Expect meta.yml to have list of categories with 'color'
    colors = []
    for cat in cats:
        if isinstance(cat, dict) and "color" in cat:
            colors.append(cat["color"])
        elif isinstance(cat, list) and len(cat) == 3:
            colors.append(cat)
        else:
            raise ValueError("Unexpected format in meta.yml")
    return np.array(colors, dtype=np.uint8)


def overlay_mask_on_frame(frame, mask, colormap, alpha=0.5):
    color_mask = colormap[mask]
    overlay = cv2.addWeighted(frame, 1 - alpha, color_mask, alpha, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--model-cfg", required=True, type=str)
    parser.add_argument("--meta-yml", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model and checkpoint
    model, variant = load_model(args.checkpoint)
    model.to(device)
    model.eval()

    # Load colormap for overlay
    colormap = load_colormap(args.meta_yml)

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare video writer for side-by-side output
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height)
    )

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
            img_tensor = img_tensor / 255.0  # normalize to 0-1
            img_tensor = img_tensor.to(device)

            # Forward pass
            pred = model(img_tensor)[0]  # assuming model returns dict or tensor
            if isinstance(pred, torch.Tensor):
                pred = pred.argmax(0).cpu().numpy()
            elif isinstance(pred, dict) and "mask" in pred:
                pred = pred["mask"].argmax(0).cpu().numpy()
            else:
                raise ValueError("Unexpected model output format")

            # Overlay prediction
            overlay = overlay_mask_on_frame(frame, pred, colormap)

            # Concatenate original + overlay
            side_by_side = np.concatenate([frame, overlay], axis=1)

            # Write frame
            out.write(side_by_side)

    cap.release()
    out.release()
    print(f"Saved output video to {args.output}")


if __name__ == "__main__":
    main()
