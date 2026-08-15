# -*- coding: utf-8 -*-
"""
论文6 (MotifDiff) 补充实验：设置 C 与设置 D
=============================================
设置 C (real + CLIP-filtered synthetic tail):
  对现有 800 张 tail 合成样本用 CLIP(图-文相似度)保留 top-50%(共400张)混入训练。
设置 D (real + mid-tail synthetic):
  对 MID 组 5 个中等类(fuziwen/guwen/huwen/jiaoyewen/yuanyangwen, 148-308框)
  各生成 200 张(共1000)混入训练，检验 mid-tail 是否受益于合成。

评估完全复用 motifdiff_eval.py 的分组与 pycocotools 逻辑，确保与 A/B 可比。
运行：python paper6_extra_exp.py --stage all   (A100, device=0, 后台)
"""
import os, json, random, shutil, argparse
from pathlib import Path
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "disabled")
import numpy as np
from PIL import Image
import torch
torch.backends.cudnn.enabled = False

ROOT = Path("/LiKun/crop-detection/paper6_MotifDiff")
REAL = ROOT / "data"
SYN_TAIL = ROOT / "synthetic"          # 4 tail classes (existing 800)
SYN_TAIL_IMG = SYN_TAIL / "images"
SYN_TAIL_LBL = SYN_TAIL / "labels"
SYN_MID = ROOT / "synthetic_mid"        # 5 mid classes (new 1000)
SYN_MID_IMG = SYN_MID / "images"
SYN_MID_LBL = SYN_MID / "labels"
SYN_FILT = ROOT / "synthetic_filtered"  # CLIP top-50% of tail (400)
SYN_FILT_IMG = SYN_FILT / "images"
SYN_FILT_LBL = SYN_FILT / "labels"

NAMES = ["bajixiangwen", "baoxianghuawen", "fuziwen", "gongziwen", "gubeiwen",
         "guwen", "hewen", "huwen", "jiaoyewen", "lianhuawen", "panchangwen",
         "shouziwen", "taiyangwen", "wanziwen", "xiziwen", "yuanyangwen", "yuwen"]
NC = 17
# 与 motifdiff_eval.py 完全一致的分组，保证与 A/B 可比
HEAD_IDS = [0, 13, 14]
MID_IDS = [1, 2, 5, 6, 7, 8, 9, 11, 15, 16]
TAIL_IDS = [3, 4, 10, 12]

TAIL_PROMPT = {
    "gongziwen": "traditional Chinese gongzi 工-shaped geometric pattern motif, blue and white porcelain style, centered, plain white background, flat design, symmetry",
    "gubeiwen": "traditional Chinese ancient shell silver ingot yuanbao pattern motif, centered, plain white background, flat design, antique",
    "taiyangwen": "traditional Chinese sun solar pattern motif with rays, centered, plain white background, flat design, red and gold",
    "panchangwen": "traditional Chinese panchang endless knot Buddhist symbol motif, interlaced lines, centered, plain white background, flat design",
}
MID = {
    "fuziwen":    dict(id=2,  n=200, prompt="traditional Chinese fu good-fortune character pattern motif, red and gold, centered, plain white background, flat design"),
    "guwen":      dict(id=5,  n=200, prompt="traditional Chinese drum pattern motif, rounded, centered, plain white background, flat design"),
    "huwen":      dict(id=7,  n=200, prompt="traditional Chinese meander key geometric pattern motif, centered, plain white background, flat design"),
    "jiaoyewen":  dict(id=8,  n=200, prompt="traditional Chinese banana leaf pattern motif, green, centered, plain white background, flat design"),
    "yuanyangwen":dict(id=15, n=200, prompt="traditional Chinese mandarin duck yuan-yang pattern motif, pair, centered, plain white background, flat design"),
}


def foreground_bbox(img, pad=0.06):
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    white = (arr[:, :, 0] > 235) & (arr[:, :, 1] > 235) & (arr[:, :, 2] > 235)
    fg = ~white
    ys, xs = np.where(fg)
    h, w = arr.shape[:2]
    if len(xs) < 50:
        return [0.5, 0.5, 0.7, 0.7]
    x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
    x0 -= int(pad * w); x1 += int(pad * w); y0 -= int(pad * h); y1 += int(pad * h)
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(w - 1, x1); y1 = min(h - 1, y1)
    cx = (x0 + x1) / 2 / w; cy = (y0 + y1) / 2 / h
    bw = (x1 - x0) / w; bh = (y1 - y0) / h
    return [round(cx, 5), round(cy, 5), round(bw, 5), round(bh, 5)]


def gen_mid():
    from diffusers import StableDiffusionXLPipeline
    repo = str(ROOT / "sdxl_ms")
    print("[GEN_MID] loading SDXL from", repo)
    sd = StableDiffusionXLPipeline.from_pretrained(
        repo, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to("cuda")
    sd.set_progress_bar_config(disable=True)
    sd.enable_attention_slicing()
    cnt = {}
    for name, cfg in MID.items():
        out_imgs = SYN_MID_IMG / name; out_lbl = SYN_MID_LBL / name
        out_imgs.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)
        made = 0; attempt = 0
        while made < cfg["n"]:
            attempt += 1
            if attempt > cfg["n"] * 3:
                break
            seed = random.randint(0, 1 << 31)
            try:
                gen = sd(prompt=cfg["prompt"], height=1024, width=1024,
                         num_images_per_prompt=1, num_inference_steps=30,
                         guidance_scale=7.5,
                         generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
            except Exception as e:
                print(f"[GEN_MID] {name} seed {seed} failed: {e}")
                continue
            gen = gen.resize((512, 512))
            bb = foreground_bbox(gen)
            if bb[2] < 0.15 or bb[3] < 0.15:
                continue
            fn = f"{name}_{made:04d}.png"
            gen.save(out_imgs / fn)
            with open(out_lbl / f"{name}_{made:04d}.txt", "w", encoding="utf-8") as f:
                f.write(f"{cfg['id']} {' '.join(str(v) for v in bb)}\n")
            made += 1
        cnt[name] = made
        print(f"[GEN_MID] {name}: {made} synthetic images")
    json.dump(cnt, open(ROOT / "gen_mid_count.json", "w"), indent=2)
    print("[GEN_MID] done:", cnt)


def clip_filter(top_ratio=0.5):
    import clip
    model, pre = clip.load("ViT-B/32", device="cuda")
    model.eval()
    SYN_FILT.mkdir(parents=True, exist_ok=True)
    total_kept = 0
    for name in TAIL_PROMPT:
        imgs_dir = SYN_TAIL_IMG / name
        lbl_dir = SYN_TAIL_LBL / name
        fns = sorted(p for p in os.listdir(imgs_dir) if p.endswith(".png"))
        if not fns:
            continue
        txt = clip.tokenize(TAIL_PROMPT[name]).to("cuda")
        scores = []
        for fn in fns:
            im = pre(Image.open(imgs_dir / fn).convert("RGB")).unsqueeze(0).to("cuda")
            with torch.no_grad():
                sim = model(im, txt)[0][0][0].item()
            scores.append((fn, sim))
        scores.sort(key=lambda x: -x[1])
        keep = max(1, int(len(scores) * top_ratio))
        sel = scores[:keep]
        odir = SYN_FILT_IMG / name
        odir.mkdir(parents=True, exist_ok=True)
        (SYN_FILT_LBL / name).mkdir(parents=True, exist_ok=True)
        for fn, _ in sel:
            shutil.copy(imgs_dir / fn, odir / fn)
            lb = lbl_dir / (fn[:-4] + ".txt")
            if lb.exists():
                shutil.copy(lb, SYN_FILT_LBL / name / (fn[:-4] + ".txt"))
        total_kept += len(sel)
        print(f"[CLIP] {name}: kept {len(sel)}/{len(scores)} (top {int(top_ratio*100)}%)")
    print(f"[CLIP] total kept: {total_kept}")


def build_train_list(tag, syn_dirs):
    real = REAL / "images/train.txt"
    lines = [l.strip() for l in open(real, encoding="utf-8") if l.strip()]
    n_syn = 0
    for d in syn_dirs:
        if not d.exists():
            continue
        for fn in sorted(d.rglob("*.png")):
            lines.append(str(fn)); n_syn += 1
    out = REAL / f"images/train_{tag}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    yaml_p = REAL / f"pattern_{tag}.yaml"
    with open(yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {REAL}\n")
        f.write(f"train: images/train_{tag}.txt\n")
        f.write("val: images/val.txt\ntest: images/test.txt\n")
        f.write("nc: 17\nnames:\n")
        for i, nm in enumerate(NAMES):
            f.write(f"  - {nm}\n")
    print(f"[YAML {tag}] train={len(lines)} (real+syn {n_syn}) -> {yaml_p}")
    return yaml_p


def train_model(tag, yaml_p, project_name):
    from ultralytics import YOLO
    m = YOLO("yolov8s.pt")
    m.train(data=str(yaml_p), epochs=80, imgsz=640, batch=32, device=0,
            name=project_name, project=str(ROOT / "runs_extra"),
            close_mosaic=10, patience=20, workers=8, optimizer="auto",
            seed=42, amp=True, cache=False, exist_ok=True)
    best = str(ROOT / "runs_extra" / project_name / "weights" / "best.pt")
    return best


# ---------- pycocotools evaluation (parametrized, from motifdiff_eval.py) ----------
def img_size(stem):
    for ext in (".jpg", ".png", ".jpeg"):
        ip = str(REAL / "images/test" / (stem + ext))
        if os.path.exists(ip):
            with Image.open(ip) as im:
                return im.size
    return None


def build_gt():
    anns, n_gt = [], [0] * NC
    lab = REAL / "labels/test"
    for f in sorted(os.listdir(lab)):
        if not f.endswith(".txt"):
            continue
        stem = f[:-4]
        sz = img_size(stem)
        if sz is None:
            continue
        W, H = sz
        for line in open(lab / f, encoding="utf-8"):
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            bw, bh = w * W, h * H
            area_ratio = w * h
            anns.append({"image_id": stem, "category_id": c,
                         "bbox": [x1, y1, bw, bh], "area": bw * bh,
                         "area_ratio": area_ratio, "iscrowd": 0})
            n_gt[c] += 1
    return anns, n_gt


def predict(model):
    import glob
    preds = []
    for imf in sorted(glob.glob(str(REAL / "images/test" / "*.*"))):
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
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
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
    return float(ev.stats[1]), float(ev.stats[0])


def evaluate(weights, out_json, model_desc):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO
    model = YOLO(weights)
    anns, n_gt = build_gt()
    preds = predict(model)
    m50, m5095 = run_eval(anns, preds)
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
    prec = ev.eval["precision"]
    ap50 = np.mean(prec[0, :, :, 0, 2], axis=0)
    ap5095 = np.mean(prec[:, :, :, 0, 2], axis=(0, 1))
    per_class = {}
    for i in range(NC):
        per_class[NAMES[i]] = {
            "ap50": (0.0 if n_gt[i] == 0 else float(ap50[i])),
            "ap5095": (0.0 if n_gt[i] == 0 else float(ap5095[i])),
            "n_gt": n_gt[i],
        }

    def grp(ids):
        return {"ap50": float(np.mean([per_class[NAMES[i]]["ap50"] for i in ids])),
                "ap5095": float(np.mean([per_class[NAMES[i]]["ap5095"] for i in ids]))}
    head, mid, tail = grp(HEAD_IDS), grp(MID_IDS), grp(TAIL_IDS)
    buckets = {"<1%": (0, 0.01), "1-5%": (0.01, 0.05),
               "5-10%": (0.05, 0.10), ">10%": (0.10, 1.01)}
    ab = {}
    for lbl, (lo, hi) in buckets.items():
        gsub = [a for a in anns if lo <= a["area_ratio"] < hi]
        img_set = set(a["image_id"] for a in gsub)
        psub = [p for p in preds if p["image_id"] in img_set]
        b50, b95 = run_eval(gsub, psub)
        ab[lbl] = {"ap50": b50, "ap5095": b95, "n_gt": len(gsub)}
    out = {"model": model_desc, "n_images": len(img_ids),
           "map50": m50, "map5095": m5095,
           "head": head, "mid": mid, "tail": tail,
           "area_buckets": ab, "per_class": per_class}
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[EVAL {model_desc}] mAP50={m50:.4f} head={head['ap50']:.4f} "
          f"mid={mid['ap50']:.4f} tail={tail['ap50']:.4f}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gen_mid", "clip_filter", "train_c", "train_d",
                             "eval_c", "eval_d", "all"])
    ap.add_argument("--top_ratio", type=float, default=0.5)
    args = ap.parse_args()

    if args.stage in ("gen_mid", "all"):
        gen_mid()
    if args.stage in ("clip_filter", "all"):
        clip_filter(top_ratio=args.top_ratio)

    if args.stage in ("train_c", "all"):
        yaml_c = build_train_list("C", [SYN_FILT_IMG])
        best_c = train_model("C", yaml_c, "extra_C")
        evaluate(best_c, str(ROOT / "c_eval.json"),
                 "YOLOv8-s (CTPD real + 400 CLIP-filtered tail synthetic)")
    if args.stage in ("train_d", "all"):
        yaml_d = build_train_list("D", [SYN_MID_IMG])
        best_d = train_model("D", yaml_d, "extra_D")
        evaluate(best_d, str(ROOT / "d_eval.json"),
                 "YOLOv8-s (CTPD real + 1000 mid-tail synthetic)")


if __name__ == "__main__":
    main()
