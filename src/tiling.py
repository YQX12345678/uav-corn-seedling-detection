"""Image tiling for UAV small-object detection.

UAV corn seedling images are high resolution (e.g. 4000x3000) with tiny
objects (~10px). Directly resizing to 640 destroys them. We instead:

    Training:  cut each image into `tile_size` tiles with `overlap` and
               keep only tiles that contain at least one object.
    Inference: slide a `tile_size` window with `overlap`; run the detector
               on each tile; map predictions back to the original image
               coordinates; run class-wise NMS across the whole image.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    x0: int
    y0: int
    x1: int
    y1: int


def slide_windows(width: int, height: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """Return list of (x0, y0, x1, y1) covering the image."""
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("invalid tile_size / overlap")
    stride = tile_size - overlap
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    wins = []
    for y in ys:
        for x in xs:
            wins.append((x, y, x + tile_size, y + tile_size))
    return wins


def cut_tiles(image: np.ndarray, tile_size: int, overlap: int) -> List[Tile]:
    h, w = image.shape[:2]
    out = []
    for (x0, y0, x1, y1) in slide_windows(w, h, tile_size, overlap):
        out.append(Tile(image=image[y0:y1, x0:x1], x0=x0, y0=y0, x1=x1, y1=y1))
    return out


def clip_box_to_tile(box_xyxy: np.ndarray, x0: int, y0: int, x1: int, y1: int):
    """Return tile-local boxes that are fully or partially inside the tile.

    Args:
        box_xyxy: (N, 4) in original-image pixel coords
    Returns:
        local: (M, 4) in tile-local coords
        keep_idx: (M,) indices into the original box array
    """
    if len(box_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    b = box_xyxy.copy().astype(np.float32)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], x0, x1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], y0, y1)
    valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    local = b[idx].copy()
    local[:, [0, 2]] -= x0
    local[:, [1, 3]] -= y0
    return local, idx


def merge_tile_predictions(tiles_preds: List[dict], class_agnostic: bool = False,
                          iou_thr: float = 0.5, score_thr: float = 0.25) -> dict:
    """Map each tile-local prediction back to global coords, then NMS.

    Args:
        tiles_preds: list of dict with keys
            'x0', 'y0', 'x1', 'y1' (tile offset)
            'boxes': (N, 4) tile-local xyxy
            'scores': (N,)
            'labels': (N,) int
    Returns:
        dict with 'boxes', 'scores', 'labels' in *global* coords, NMS applied.
    """
    all_boxes, all_scores, all_labels = [], [], []
    for t in tiles_preds:
        if len(t["boxes"]) == 0:
            continue
        b = t["boxes"].astype(np.float32).copy()
        b[:, [0, 2]] += t["x0"]
        b[:, [1, 3]] += t["y0"]
        keep = t["scores"] >= score_thr
        all_boxes.append(b[keep])
        all_scores.append(t["scores"][keep])
        all_labels.append(t["labels"][keep])
    if not all_boxes:
        return {"boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64)}
    boxes = np.concatenate(all_boxes, 0)
    scores = np.concatenate(all_scores, 0)
    labels = np.concatenate(all_labels, 0)
    return nms_per_class(boxes, scores, labels, iou_thr=iou_thr, class_agnostic=class_agnostic)


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                  iou_thr: float = 0.5, class_agnostic: bool = False) -> dict:
    """Pure numpy NMS, class-wise by default."""
    if len(boxes) == 0:
        return {"boxes": boxes, "scores": scores, "labels": labels}
    order = np.argsort(-scores)
    boxes, scores, labels = boxes[order], scores[order], labels[order]
    keep = []
    if class_agnostic:
        return _greedy_nms(boxes, scores, labels, np.ones(len(boxes), dtype=bool), iou_thr)
    out_boxes, out_scores, out_labels = [], [], []
    for c in np.unique(labels):
        mask = labels == c
        sub = _greedy_nms(boxes[mask], scores[mask], labels[mask], mask, iou_thr)
        out_boxes.append(sub["boxes"]); out_scores.append(sub["scores"]); out_labels.append(sub["labels"])
    boxes = np.concatenate(out_boxes, 0)
    scores = np.concatenate(out_scores, 0)
    labels = np.concatenate(out_labels, 0)
    order = np.argsort(-scores)
    return {"boxes": boxes[order], "scores": scores[order], "labels": labels[order]}


def _greedy_nms(boxes, scores, labels, keep_mask, iou_thr):
    idxs = np.arange(len(boxes))
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = _iou_vec(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_thr]
    keep = np.array(keep, dtype=np.int64)
    return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


def _iou_vec(box, others):
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    b = (others[:, 2] - others[:, 0]).clip(0) * (others[:, 3] - others[:, 1]).clip(0)
    return inter / np.maximum(a + b - inter, 1e-9)
