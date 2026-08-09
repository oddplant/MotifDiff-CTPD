"""
论文6 基线 A：纯真实数据训练 YOLOv8-s（不含 MotifDiff 合成样本）
=================================================================
目的：与「真实+合成」(motifdiff) 对比，量化合成数据的增益。
数据：CTPD 官方 train/val/test（pattern_data.yaml），不合并 synthetic。
训练完用 motifdiff_eval.py --weights <best> --out baseline_eval.json 评测。
"""
import torch
torch.backends.cudnn.enabled = False  # A800 env cuDNN fix
from ultralytics import YOLO
from pathlib import Path

ROOT = Path("/LiKun/crop-detection/paper6_MotifDiff")
REAL = ROOT / "data"


def main():
    m = YOLO("yolov8s.pt")
    m.train(
        data=str(REAL / "pattern_data.yaml"),
        epochs=80,
        imgsz=640,
        batch=32,
        device=0,
        project=str(ROOT / "runs_baseline"),
        name="baseline",
        exist_ok=True,
        seed=42,
    )
    print("[BASELINE] training done")


if __name__ == "__main__":
    main()
