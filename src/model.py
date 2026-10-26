"""Detectors. Ultralytics YOLOv8 is the competition path; TinyDetector is the
pure-PyTorch fallback for CPU smoke tests."""
import torch
import torch.nn as nn
import torchvision.models as models


class TinyDetector(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(*list(bb.children())[:-2])
        self.head = nn.Conv2d(512, num_anchors * (5 + num_classes), 1)

    def forward(self, x):
        f = self.stem(x)
        return self.head(f)


def build_ultralytics_model(weights="yolov8n.pt", num_classes=1):
    from ultralytics import YOLO
    model = YOLO(weights)
    model.model.nc = num_classes
    return model
