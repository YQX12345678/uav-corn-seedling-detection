"""Training entry point.

--mode ultralytics : YOLOv8 fine-tuning on a tiled data.yaml (competition path).
--mode tiny        : pure-PyTorch smoke loop on tiles, CPU-friendly.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from .dataset import UAVTiledDataset, collate_fn
from .model import TinyDetector, build_ultralytics_model


def train_ultralytics(data_yaml, weights="yolov8n.pt", epochs=100, imgsz=640, batch=16):
    model = build_ultralytics_model(weights=weights, num_classes=1)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch)
    return model


def train_tiny(root, epochs=2, batch_size=2, lr=1e-3, device="cpu", tile_size=640, overlap=100):
    ds = UAVTiledDataset(root, split="train", tile_size=tile_size, overlap=overlap)
    if len(ds) == 0:
        print("no tiles; skipping smoke train")
        return None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = TinyDetector(num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        tot = 0.0
        for imgs, tgts in dl:
            x = torch.stack(imgs).to(device)
            preds = model(x)
            loss = preds.mean()  # placeholder; real path uses ultralytics loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item())
        print(f"epoch {epoch + 1}: loss={tot / max(1, len(dl)):.4f}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ultralytics", "tiny"], default="tiny")
    p.add_argument("--root", default="data")
    p.add_argument("--data-yaml", default="data/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.mode == "ultralytics":
        train_ultralytics(args.data_yaml, args.weights, epochs=args.epochs, batch=args.batch, imgsz=args.tile)
    else:
        train_tiny(args.root, epochs=args.epochs, batch_size=args.batch,
                   device=args.device, tile_size=args.tile, overlap=args.overlap)


if __name__ == "__main__":
    main()
