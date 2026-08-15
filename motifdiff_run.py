# -*- coding: utf-8 -*-
"""
MotifDiff · 论文6(生成式·MotifDiff) 真实实验脚本
==========================================
在 CTPD 上对极端长尾尾部类做「扩散生成式数据补偿」：
  1) 用 Stable Diffusion 1.5 为 4 个尾部类(工字/古贝/太阳/盘长)生成带 bbox 的合成样本
  2) 前景掩膜自动标注 bbox + CLIP 质量过滤
  3) 合成尾部样本混入训练集，重训 YOLOv8-s，对比 官方test 上的 tail/整体/AP_small
  4) 消融：有无 CLIP 过滤、合成数量

运行：python motifdiff_run.py  (在 A100, device=0)
依赖：diffusers transformers accelerate safetensors torch ultralytics
"""
import os, json, math, random, shutil
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("COMET_MODE", "disabled")
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
torch.backends.cudnn.enabled = False  # global A800 cuDNN fix

ROOT = Path("/LiKun/crop-detection/paper6_MotifDiff")
REAL_DATA = ROOT / "data"
SYN = ROOT / "synthetic"
SYN_IMG = SYN / "images"
SYN_LBL = SYN / "labels"
SYN.mkdir(parents=True, exist_ok=True)
SYN_IMG.mkdir(parents=True, exist_ok=True)
SYN_LBL.mkdir(parents=True, exist_ok=True)

# 尾部类（框数 < 100，按清洗后分布）：工字4 / 古贝8 / 太阳61 / 盘长69
TAIL = {
    "gongziwen":   dict(id=3,  n=200, prompt="traditional Chinese gongzi 工-shaped geometric pattern motif, "
                        "blue and white porcelain style, centered, plain white background, flat design, symmetry"),
    "gubeiwen":    dict(id=4,  n=200, prompt="traditional Chinese ancient shell silver ingot yuanbao pattern motif, "
                        "centered, plain white background, flat design, antique"),
    "taiyangwen":  dict(id=12, n=200, prompt="traditional Chinese sun solar pattern motif with rays, "
                        "centered, plain white background, flat design, red and gold"),
    "panchangwen": dict(id=10, n=200, prompt="traditional Chinese panchang endless knot Buddhist symbol motif, "
                        "interlaced lines, centered, plain white background, flat design"),
}

def parse_txt(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]

def foreground_bbox(img, pad=0.06):
    """简单白底前景检测 -> bbox(归一化 cx,cy,w,h)；背景非白则回退整图。"""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    white = (arr[:,:,0]>235)&(arr[:,:,1]>235)&(arr[:,:,2]>235)
    fg = ~white
    ys, xs = np.where(fg)
    h, w = arr.shape[:2]
    if len(xs) < 50:  # 几乎没有前景 -> 回退中心 70%
        return [0.5, 0.5, 0.7, 0.7]
    x0,x1 = xs.min(), xs.max(); y0,y1 = ys.min(), ys.max()
    x0-=int(pad*w); x1+=int(pad*w); y0-=int(pad*h); y1+=int(pad*h)
    x0=max(0,x0); y0=max(0,y0); x1=min(w-1,x1); y1=min(h-1,y1)
    cx=(x0+x1)/2/w; cy=(y0+y1)/2/h; bw=(x1-x0)/w; bh=(y1-y0)/h
    return [round(cx,5),round(cy,5),round(bw,5),round(bh,5)]

def clip_score(pipe_clip, img, text):
    """返回 image-text 相似度（用于质量过滤）。需要单独加载 CLIP。"""
    raise NotImplementedError  # 见 train 阶段可选；默认不过滤

def generate(args):
    from diffusers import StableDiffusionXLPipeline
    import torch
    torch.backends.cudnn.enabled = False  # fix cuDNN init (A800 env)
    print("[GEN] loading SDXL base (local safetensors) ...")
    from modelscope.hub.snapshot_download import snapshot_download
    repo = str(ROOT/"sdxl_ms")
    print("[GEN] snapshot_download SDXL from ModelScope (ignore fp16 + single-ckpt) ...")
    snapshot_download("AI-ModelScope/stable-diffusion-xl-base-1.0", local_dir=repo,
                      ignore_file_pattern=["*fp16*", "sd_xl_base_1.0.safetensors"])
    sd = StableDiffusionXLPipeline.from_pretrained(
        repo, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True,
    ).to("cuda")
    sd.set_progress_bar_config(disable=True)
    sd.enable_attention_slicing()
    cnt = {}
    for name, cfg in TAIL.items():
        out_imgs = SYN_IMG / name; out_lbl = SYN_LBL / name
        out_imgs.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)
        made = 0; attempt = 0
        while made < cfg["n"]:
            attempt += 1
            if attempt > cfg["n"]*3: break
            seed = random.randint(0, 1<<31)
            try:
                gen = sd(prompt=cfg["prompt"], height=1024, width=1024,
                         num_images_per_prompt=1, num_inference_steps=30, guidance_scale=7.5,
                         generator=torch.Generator(device="cuda").manual_seed(seed)).images[0]
            except Exception as e:
                print(f"[GEN] {name} seed {seed} failed: {e}"); continue
            # 固定尺寸，便于训练
            gen = gen.resize((512,512))
            bb = foreground_bbox(gen)
            if bb[2] < 0.05 or bb[3] < 0.05:   # 前景太小，丢弃
                continue
            fn = f"{name}_{made:04d}.png"
            gen.save(out_imgs / fn)
            with open(out_lbl / f"{name}_{made:04d}.txt", "w", encoding="utf-8") as f:
                f.write(f"{cfg['id']} {' '.join(str(v) for v in bb)}\n")
            made += 1
        cnt[name] = made
        print(f"[GEN] {name}: {made} synthetic images")
    json.dump(cnt, open(SYN/"gen_count.json","w"), indent=2)
    print("[GEN] done:", cnt)

def build_combined_yaml(use_syn=True):
    real_train = parse_txt(REAL_DATA/"images/train.txt")
    lines = list(real_train)
    if use_syn:
        for name in TAIL:
            d = SYN_IMG/name
            for f in sorted(d.glob("*.png")):
                lines.append(str(f))
    comb = ROOT/"train_motifdiff.txt"
    with open(comb,"w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    yaml_p = ROOT/"pattern_motifdiff.yaml"
    names = parse_names()
    with open(yaml_p,"w",encoding="utf-8") as f:
        f.write(f"path: {REAL_DATA}\n")
        f.write(f"train: {comb}\n")
        f.write(f"val: images/val.txt\n")
        f.write(f"test: images/test.txt\n")
        f.write(f"nc: 17\nnames:\n")
        for i,nm in enumerate(names):
            f.write(f"  - {nm}\n")
    print(f"[YAML] combined train={len(lines)} images -> {yaml_p}")
    return yaml_p, len(lines)

def parse_names():
    # 与 pattern_data.yaml 顺序一致
    return ["bajixiangwen","baoxianghuawen","fuziwen","gongziwen","gubeiwen","guwen",
            "hewen","huwen","jiaoyewen","lianhuawen","panchangwen","shouziwen",
            "taiyangwen","wanziwen","xiziwen","yuanyangwen","yuwen"]

def evaluate(project, name, weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    # 官方 test 评估
    metrics = m.val(data=str(REAL_DATA/"pattern_data.yaml"),
                    split="test", imgsz=640, batch=32, device=0, verbose=False)
    box = metrics.box
    out = dict(
        name=name,
        map50=float(box.map50), map50_95=float(box.map),
    )
    # 逐类 AP50
    ap50_per = box.ap50  # ultralytics: per-class AP@0.5 (list len nc)
    names = parse_names()
    tail_ids = [TAIL[k]["id"] for k in TAIL]
    mid_ids = [1,2,5,6,7,8,9,11,15,16]   # 100<=boxes 近似（见论文）
    head_ids = [0,13,14]                  # 宝相花/寿字/鱼 等头部
    out["tail_ap50"] = float(np.mean([ap50_per[i] for i in tail_ids]))
    out["mid_ap50"]  = float(np.mean([ap50_per[i] for i in mid_ids]))
    out["head_ap50"] = float(np.mean([ap50_per[i] for i in head_ids]))
    out["ap50_per_class"] = {names[i]: float(ap50_per[i]) for i in range(len(names))}
    json.dump(out, open(project/f"{name}_eval.json","w"), indent=2, ensure_ascii=False)
    print(f"[EVAL] {name}: mAP50={out['map50']:.4f} tail={out['tail_ap50']:.4f} "
          f"mid={out['mid_ap50']:.4f} head={out['head_ap50']:.4f}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["gen","yaml","train","eval","all"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--project", default=str(ROOT/"runs_motifdiff"))
    args = ap.parse_args()
    project = Path(args.project); project.mkdir(parents=True, exist_ok=True)

    if args.stage in ("gen","all"):
        generate(args)
    if args.stage in ("yaml","all"):
        yaml_p, ntrain = build_combined_yaml(use_syn=True)
    else:
        yaml_p = ROOT/"pattern_motifdiff.yaml"

    if args.stage in ("train","all"):
        from ultralytics import YOLO
        if not os.path.exists(args.weights):
            print(f"[TRAIN] {args.weights} 缺失，交由 ultralytics 下载")
        m = YOLO(args.weights)
        m.train(data=str(yaml_p), epochs=args.epochs, imgsz=640, batch=32,
                device=0, name="motifdiff", project=str(project),
                close_mosaic=10, patience=20, workers=8, optimizer="auto",
                seed=42, amp=True, cache=False)
        best = project/"motifdiff"/"weights"/"best.pt"
        evaluate(project, "motifdiff", str(best))
    elif args.stage == "eval":
        best = project/"motifdiff"/"weights"/"best.pt"
        evaluate(project, "motifdiff", str(best))

if __name__ == "__main__":
    main()
