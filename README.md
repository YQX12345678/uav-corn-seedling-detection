# iFLYTEK UAV Corn Seedling Detection

**无人机航拍玉米苗检测：小目标切片训练 + 重叠推理 + 跨切片 NMS 合并。YOLOv8 主路径 + 纯 PyTorch fallback。**

## 1. 赛题背景

无人机航拍图像分辨率高（常见 4000×3000），玉米苗单株只有 10–20 像素。**直接缩放到 640 会让目标消失**，这是本赛题区别于普通目标检测的核心难点。评测指标为 mAP@0.5。

## 2. 方案核心：切片化

| 阶段 | 做法 |
|---|---|
| 训练 | 把每张大图切成 `tile_size=640`、`overlap=100` 的切片；只保留至少含 1 个目标的切片；背景切片全部丢弃 |
| 推理 | 同样滑窗切片；每片单独送进 YOLOv8；切片内的预测映射回原图坐标 |
| 合并 | **逐类别 NMS** 合并重叠区域的重复框；`iou=0.45` 起调 |
| 计数 | 提供 `count_metrics`（MAE / RMSE），对齐计数类评分 |

## 3. 模块一览

| 模块 | 内容 |
|---|---|
| `src/tiling.py` | `cut_tiles`、`clip_box_to_tile`、`nms_per_class`、`merge_tile_predictions`（纯 numpy） |
| `src/dataset.py` | `UAVTiledDataset`：每张图按切片展开为多个样本；tiny box 过滤 |
| `src/model.py` | `TinyDetector`（ResNet18 + anchor head）与 `build_ultralytics_model` |
| `src/train.py` | `--mode ultralytics`（比赛主路径）或 `--mode tiny`（CPU smoke） |
| `src/infer.py` | `run_tiled_ultralytics` / `run_tiled_jsonl`：切片推理 + 跨片 NMS + 提交文件 |
| `src/evaluate.py` | VOC 11-point mAP@0.5 + 计数 MAE/RMSE |
| `scripts/prepare_data.py` | VOC → YOLO 格式转换，生成 `data.yaml` |

## 4. 训练流程（比赛路径）

    # 1) 数据整理
    python scripts/prepare_data.py --voc-root data/VOCdevkit --out data

    # 2) 训练 YOLOv8（数据已经是切片形式；训练时 imgsz=tile_size）
    python -m src.train --mode ultralytics --data-yaml data/data.yaml \
        --weights yolov8n.pt --epochs 100 --batch 16 --tile 640

    # 3) 切片推理 + 跨片 NMS
    python -c "from src.infer import run_tiled_ultralytics; run_tiled_ultralytics( \
        'runs/detect/train/weights/best.pt', 'data/images/val', 'submission.csv', \
        tile_size=640, overlap=100)"

## 5. Fallback（无 ultralytics 依赖）

    python -m src.train --mode tiny --root data --epochs 2 --batch 2 --tile 640 --overlap 100

## 6. 关键工程决策

- **为什么切片而不是缩放？** UAV 苗检测的召回瓶颈在极小目标，切片让每个目标在送进网络时仍保持 10px+ 的可辨识尺度。
- **为什么只保留正样本切片？** 背景切片占总切片数 80%+ 且不提供训练信号，过滤后训练时间骤降，且每步都能算损失。
- **为什么逐类别 NMS？** 跨切片合并时类别边界会互相压制，逐类别 NMS 避免误删。
- **为什么同时保留 CSV/JSONL 两种提交？** 讯飞各 track 提交格式不完全统一，两种生成器可切换。
- **训练数据切片 + 推理切片尺寸一致**：tile_size 在训练和推理必须一致（默认 640），否则特征尺度不匹配。

## 7. 文件结构

    src/tiling.py     切片、坐标映射、逐类 NMS
    src/dataset.py    切片化数据集
    src/model.py      TinyDetector / YOLOv8 wrapper
    src/train.py      训练入口
    src/infer.py      切片推理 + 提交
    src/evaluate.py   mAP@0.5 + MAE/RMSE
    scripts/prepare_data.py
    configs/default.yaml
    requirements.txt
    LICENSE
    .gitignore

## 8. 我的贡献

- 独立实现切片化训练/推理流水线（`tiling.py`），覆盖坐标映射、重叠裁剪、逐类 NMS。
- 实现切片化数据集，自动只保留正样本切片。
- 实现跨切片 NMS 合并与 CSV/JSONL 两种提交格式。
- 同时实现检测 mAP@0.5 与计数 MAE/RMSE，覆盖讯飞两种常见评分维度。
