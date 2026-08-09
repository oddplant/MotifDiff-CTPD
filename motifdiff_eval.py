"""
论文6 (MotifDiff) 官方 test 集评估  ——  pycocotools 自算版
=================================================================
解决原 motifdiff_run.py evaluate() 的 IndexError：
  ultralytics box.ap50 只返回 test 集「有 GT」的类（CTPD 的 gubeiwen
  在 test 中 n_gt=0，被 COCOeval 剔除），长度 16 而非 17，导致 [16] 越界。

本脚本用 pycocotools 直接算，n_gt=0 的类 AP 填 0，并产出：
  - 总体 mAP@0.5 / mAP@0.5:0.95
  - 逐类 AP50 / AP50-95 / n_gt
  - head / mid / tail 分组（与论文5 同一组 ID，跨论文一致）
  - 面积桶（按 bbox 占画幅比例 <1% / 1-5% / 5-10% / >10%，与论文5 一致）
"""
import os, glob, json
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import torch
torch.backends.cudnn.enabled = False  # A800 env cuDNN fix
from ultralytics import YOLO

ROOT = Path("/LiKun/crop-detection/paper6_MotifDiff")
REAL = ROOT / "data"
IMG_DIR = str(REAL / "images/test")
LAB_DIR = str(REAL / "labels/test")
NAMES = ["bajixiangwen", "baoxianghuawen", "fuziwen", "gongziwen", "gubeiwen", "guwen",
         "hewen", "huwen", "jiaoyewen", "lianhuawen", "panchangwen", "shouziwen",
         "taiyangwen", "wanziwen", "xiziwen", "yuanyangwen", "yuwen"]
NC = 17
# 与论文5 一致的长尾分组（按训练集 boxes 数近似）
HEAD_IDS = [0, 13, 14]
MID_IDS  = [1, 2, 5, 6, 7, 8, 9, 11, 15, 16]
TAIL_IDS = [3, 4, 10, 12]
BEST = str(ROOT / "runs_motifdiff/motifdiff/weights/best.pt")


def img_size(stem):
    for ext in (".jpg", ".png", ".jpeg"):
        ip = os.path.join(IMG_DIR, stem + ext)
        if os.path.exists(ip):
            with Image.open(ip) as im:
                return im.size  # (W, H)
    return None


def build_gt():
    """YOLO test labels -> COCO anns；记录每类 n_gt 与每个 ann 的画幅占比 area_ratio。"""
    anns, n_gt, sizes = [], [0] * NC, {}
    aid = 1
    for f in sorted(os.listdir(LAB_DIR)):
        if not f.endswith(".txt"):
            continue
        stem = f[:-4]
        sz = img_size(stem)
        if sz is None:
            continue
        W, H = sz
        for line in open(os.path.join(LAB_DIR, f), encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            bw, bh = w * W, h * H
            area_ratio = w * h  # 占画幅比例（归一化）
            anns.append({"image_id": stem, "category_id": c,
                         "bbox": [x1, y1, bw, bh], "area": bw * bh,
                         "area_ratio": area_ratio, "iscrowd": 0, "id": aid})
            aid += 1; n_gt[c] += 1
        sizes[stem] = sz
    return anns, n_gt, sizes


def predict(model):
    preds, imgs = [], sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    for imf in imgs:
        stem = os.path.splitext(os.path.basename(imf))[0]
        res = model.predict(source=imf, conf=0.001, iou=0.7,
                             device=0, verbose=False, imgsz=640)[0]
        if res.boxes is None:
            continue
        for b in range(len(res.boxes)):
            cls = int(res.boxes.cls[b]); conf = float(res.boxes.conf[b])
            x1, y1, x2, y2 = res.boxes.xyxy[b].tolist()
            preds.append({"image_id": stem, "category_id": cls,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": conf})
    return preds


def run_eval(gt_sub, preds_sub):
    """对给定 GT 子集跑一次 COCOeval，返回 (mAP50, mAP50-95)。"""
    if not gt_sub:
        return 0.0, 0.0
    img_ids = sorted(set(a["image_id"] for a in gt_sub))
    imgs = [{"id": i} for i in img_ids]
    gt2 = [dict(a) for a in gt_sub]
    for ai, a in enumerate(gt2):
        a["id"] = ai + 1; a.pop("area_ratio", None)
    cocoGt = COCO()
    cocoGt.dataset = {"images": imgs, "annotations": gt2,
                      "categories": [{"id": i} for i in range(NC)]}
    cocoGt.createIndex()
    preds_s = [p for p in preds_sub if p["image_id"] in set(img_ids)]
    cocoDt = cocoGt.loadRes(preds_s)
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    ev.params.iouThrs = np.linspace(0.5, 0.95, 10)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])  # mAP50, mAP50-95


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=BEST, help="训练好的模型权重")
    ap.add_argument("--out", default=str(ROOT / "motifdiff_eval.json"), help="输出 json")
    args = ap.parse_args()
    model = YOLO(args.weights)
    anns, n_gt, _ = build_gt()
    preds = predict(model)

    # 总体
    m50, m5095 = run_eval(anns, preds)

    # 逐类 AP（需 precision 数组）
    img_ids = sorted(set(a["image_id"] for a in anns))
    imgs = [{"id": i} for i in img_ids]
    gt2 = [dict(a) for a in anns]
    for ai, a in enumerate(gt2):
        a["id"] = ai + 1; a.pop("area_ratio", None)
    cocoGt = COCO()
    cocoGt.dataset = {"images": imgs, "annotations": gt2,
                      "categories": [{"id": i} for i in range(NC)]}
    cocoGt.createIndex()
    cocoDt = cocoGt.loadRes([p for p in preds if p["image_id"] in set(img_ids)])
    ev = COCOeval(cocoGt, cocoDt, "bbox")
    ev.params.iouThrs = np.linspace(0.5, 0.95, 10)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    prec = ev.eval["precision"]  # (10 iou, 101 rec, NC, 4 area, 3 maxDets)
    # prec shape (10 iou, 101 rec, NC, 4 area, 3 maxDets)
    # 先固定 area=0(all), maxDets=2(100)，再对 recall 维 mean，保留 NC
    ap50 = np.mean(prec[0, :, :, 0, 2], axis=0)            # (101,NC) -> (NC,)
    ap5095 = np.mean(prec[:, :, :, 0, 2], axis=(0, 1))     # (10,101,NC) -> (NC,)

    per_class = {}
    for i in range(NC):
        per_class[NAMES[i]] = {
            "ap50": (0.0 if n_gt[i] == 0 else float(ap50[i])),
            "ap5095": (0.0 if n_gt[i] == 0 else float(ap5095[i])),
            "n_gt": n_gt[i],
        }

    def grp(ids):
        a50 = np.mean([per_class[NAMES[i]]["ap50"] for i in ids])
        a95 = np.mean([per_class[NAMES[i]]["ap5095"] for i in ids])
        return {"ap50": float(a50), "ap5095": float(a95)}

    head, mid, tail = grp(HEAD_IDS), grp(MID_IDS), grp(TAIL_IDS)

    # 面积桶（占画幅比例）
    buckets = {"<1%": (0, 0.01), "1-5%": (0.01, 0.05),
               "5-10%": (0.05, 0.10), ">10%": (0.10, 1.01)}
    ab = {}
    for lbl, (lo, hi) in buckets.items():
        gsub = [a for a in anns if lo <= a["area_ratio"] < hi]
        img_set = set(a["image_id"] for a in gsub)
        psub = [p for p in preds if p["image_id"] in img_set]
        b50, b95 = run_eval(gsub, psub)
        ab[lbl] = {"ap50": b50, "ap5095": b95, "n_gt": len(gsub)}

    out = {
        "model": "YOLOv8-s (CTPD real + 800 MotifDiff synthetic)",
        "n_images": len(img_ids),
        "map50": m50, "map5095": m5095,
        "head": head, "mid": mid, "tail": tail,
        "area_buckets": ab,
        "per_class": per_class,
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
