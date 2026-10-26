"""Tiled inference + cross-tile NMS + submission writer.

This is the key module that makes UAV small-object detection work:
- Image is split into overlapping tiles
- Each tile is run through YOLOv8 at its native resolution (640x640)
- Tile-local boxes are mapped back to image coords
- Class-wise NMS merges duplicates across overlaps
"""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

from .tiling import cut_tiles, merge_tile_predictions


def run_tiled_ultralytics(weights, image_dir, output_csv,
                          tile_size=640, overlap=100, conf=0.25, iou=0.45):
    from ultralytics import YOLO
    model = YOLO(weights)
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "class_id", "x_min", "y_min", "x_max", "y_max", "score"])
        for img_path in images:
            img = cv2.imread(str(img_path))
            tiles = cut_tiles(img, tile_size, overlap)
            tile_preds = []
            for t in tiles:
                r = model.predict(t.image, imgsz=tile_size, conf=conf, iou=iou, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    tile_preds.append({"x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                                        "boxes": np.zeros((0, 4)), "scores": np.zeros((0,)),
                                        "labels": np.zeros((0,), dtype=np.int64)})
                    continue
                tile_preds.append({
                    "x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                    "boxes": r.boxes.xyxy.cpu().numpy(),
                    "scores": r.boxes.conf.cpu().numpy(),
                    "labels": r.boxes.cls.cpu().numpy().astype(np.int64),
                })
            merged = merge_tile_predictions(tile_preds, iou_thr=iou, score_thr=conf)
            for (x1, y1, x2, y2), s, c in zip(merged["boxes"], merged["scores"], merged["labels"]):
                w.writerow([img_path.stem, int(c), float(x1), float(y1), float(x2), float(y2), float(s)])
    print(f"tiled detections saved to {output_csv}")


def run_tiled_jsonl(weights, image_dir, output_jsonl,
                    tile_size=640, overlap=100, conf=0.25, iou=0.45):
    from ultralytics import YOLO
    model = YOLO(weights)
    image_dir = Path(image_dir)
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    with open(output_jsonl, "w") as f:
        for img_path in images:
            img = cv2.imread(str(img_path))
            tiles = cut_tiles(img, tile_size, overlap)
            tile_preds = []
            for t in tiles:
                r = model.predict(t.image, imgsz=tile_size, conf=conf, iou=iou, verbose=False)[0]
                if r.boxes is None:
                    continue
                tile_preds.append({
                    "x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                    "boxes": r.boxes.xyxy.cpu().numpy(),
                    "scores": r.boxes.conf.cpu().numpy(),
                    "labels": r.boxes.cls.cpu().numpy().astype(np.int64),
                })
            merged = merge_tile_predictions(tile_preds, iou_thr=iou, score_thr=conf)
            preds = [{
                "class_id": int(c),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(s),
            } for (x1, y1, x2, y2), c, s in zip(merged["boxes"], merged["labels"], merged["scores"])]
            f.write(json.dumps({"image_id": img_path.stem, "predictions": preds}) + "\n")
    print(f"tiled detections saved to {output_jsonl}")
