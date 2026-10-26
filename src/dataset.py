"""UAV corn seedling dataset with tiled training.

Directory layout:
    data/images/{train,val}/*.jpg
    data/labels/{train,val}/*.txt  # cls cx cy w h, normalized to [0,1]
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .tiling import cut_tiles, clip_box_to_tile

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = {0: "corn_seedling"}


class UAVTiledDataset(Dataset):
    """A dataset whose every item is a valid *tile* extracted from an image.

    Only tiles containing at least one object are kept, which suppresses
    background tiles and dramatically speeds up training.
    """
    def __init__(self, root, split="train", tile_size=640, overlap=100, min_box_px=2):
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_box_px = min_box_px
        self.index = []
        for img_path in sorted(self.img_dir.glob("*.jpg")) + sorted(self.img_dir.glob("*.png")):
            self.index.extend(self._enumerate_tiles(img_path))

    def _enumerate_tiles(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return []
        h, w = img.shape[:2]
        boxes_norm, labels = self._load_labels(self.lbl_dir / (img_path.stem + ".txt"))
        boxes_px = boxes_norm * np.array([w, h, w, h], dtype=np.float32)
        keep = []
        for t in cut_tiles(np.zeros((h, w, 3), np.uint8), self.tile_size, self.overlap):
            local, idx = clip_box_to_tile(boxes_px, t.x0, t.y0, t.x1, t.y1)
            if len(idx) == 0:
                continue
            # filter tiny boxes
            wpx = local[:, 2] - local[:, 0]
            hpx = local[:, 3] - local[:, 1]
            ok = (wpx >= self.min_box_px) & (hpx >= self.min_box_px)
            if not ok.any():
                continue
            keep.append({
                "img_path": str(img_path),
                "x0": t.x0, "y0": t.y0, "x1": t.x1, "y1": t.y1,
                "boxes": local[ok],
                "labels": labels[idx][ok],
            })
        return keep

    def _load_labels(self, path):
        if not path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if not lines:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        arr = np.array([ln.split() for ln in lines], dtype=np.float32)
        cls = arr[:, 0].astype(np.int64)
        cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), cls

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec = self.index[idx]
        img = cv2.imread(rec["img_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tile = img[rec["y0"]:rec["y1"], rec["x0"]:rec["x1"]]
        if tile.shape[:2] != (self.tile_size, self.tile_size):
            tile = cv2.resize(tile, (self.tile_size, self.tile_size))
        tile = tile.astype(np.float32) / 255.0
        tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(tile).permute(2, 0, 1).contiguous()

        tgt = {
            "boxes": torch.from_numpy(rec["boxes"]).float(),
            "labels": torch.from_numpy(rec["labels"]).long(),
        }
        return x, tgt


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
