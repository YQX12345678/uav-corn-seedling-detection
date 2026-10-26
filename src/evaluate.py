"""mAP@0.5 evaluation for UAV seedling detection.

Uses VOC 11-point AP. Reuses the class-wise NMS from tiling.nms_per_class to
make sure evaluations are identical to the ones the pipeline ships.
"""
import numpy as np
from collections import defaultdict

from .tiling import _iou_vec


def _voc_ap(recall, precision):
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map50(detections, ground_truth, num_classes, iou_thr=0.5):
    gt_by_image = defaultdict(lambda: defaultdict(list))
    for g in ground_truth:
        gt_by_image[g["image_id"]][g["class_id"]].append(np.array(g["bbox"], dtype=np.float32))
    n_gt = defaultdict(int)
    for img, cls_map in gt_by_image.items():
        for c, boxes in cls_map.items():
            n_gt[c] += len(boxes)

    dets_by_class = defaultdict(list)
    for d in detections:
        dets_by_class[d["class_id"]].append(d)

    aps = {}
    for c in range(num_classes):
        if n_gt[c] == 0:
            continue
        dets = sorted(dets_by_class.get(c, []), key=lambda x: -x["score"])
        used = defaultdict(set)
        tp = np.zeros(len(dets), dtype=np.float32)
        for i, d in enumerate(dets):
            gt_boxes = gt_by_image[d["image_id"]].get(c, [])
            best_iou, best_j = 0.0, -1
            b = np.array(d["bbox"], dtype=np.float32)
            for j, gb in enumerate(gt_boxes):
                if j in used[d["image_id"]]:
                    continue
                v = float(_iou_vec(b, gb[None, :])[0])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                tp[i] = 1.0
                used[d["image_id"]].add(best_j)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(1 - tp)
        recall = cum_tp / n_gt[c]
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        aps[c] = _voc_ap(recall, precision)

    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return aps, mean_ap


def count_metrics(pred_counts, gt_counts):
    """For counting tasks: MAE and RMSE per image."""
    p = np.array(pred_counts, dtype=np.float32)
    g = np.array(gt_counts, dtype=np.float32)
    if len(p) == 0:
        return {"mae": 0.0, "rmse": 0.0, "n": 0}
    mae = float(np.mean(np.abs(p - g)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(p))}
