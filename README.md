# MotifDiff-CTPD

Code release for the paper:

> **When Generative Augmentation Fails: An Empirical Diagnosis of Diffusion-Based Synthesis for Long-Tail Heritage Pattern Detection**
> Li Dong
> *Neurocomputing* (Elsevier; to be submitted)

This repository contains the synthesis, training, and evaluation code for the empirical A/B diagnosis of generative (diffusion-based) data augmentation on the **Chinese Traditional Pattern Dataset (CTPD)** — a publicly released, extreme long-tail heritage-motif detection benchmark (Li et al., 2026, Science Data Bank, DOI: 10.57760/sciencedb.34731, CC BY 4.0).

## TL;DR

We synthesized 800 tail-class samples (200 each for the four rarest motifs: *gongziwen*, *gubeiwen*, *taiyangwen*, *panchangwen*) with SDXL and mixed them into YOLOv8-s training, then compared against a strictly matched real-only baseline (Setting A) on the official 862-image test split. **Result: diffusion synthesis yields no measurable gain** (ΔmAP@0.5 = −0.003; Δtail = −0.006) and a slight regression on the very tail classes it targeted. Only *taiyangwen* improves (+0.050). See the paper for the three-factor failure-mode diagnosis (domain shift, pseudo-label noise, statistical underpowering).

## Environment

- Python 3.10+, PyTorch 2.4+, CUDA 12.1
- `ultralytics >= 8.4` (YOLOv8 / World APIs)
- `diffusers >= 0.30`, `transformers >= 4.51`
- `pycocotools`, `numpy`, `Pillow`
- SDXL weights: `stabilityai/stable-diffusion-xl-base-1.0` (we pulled via the ModelScope mirror, ~22 GB)

> Note: on the A800 node we had to set `torch.backends.cudnn.enabled = False` before inference/training to avoid a `CUDNN_STATUS_NOT_INITIALIZED` error. The scripts already include this fix.

## Files

| File | Purpose |
|------|---------|
| `motifdiff_run.py` | Main protocol: `--stage gen` (synthesize 800 samples) → `yaml` (build combined train manifest) → `train` (YOLOv8-s 80 epochs) → `eval` (official test). |
| `baseline_train.py` | Setting A: identical YOLOv8-s training on **real CTPD only** (no synthesis). |
| `motifdiff_eval.py` | Strict A/B evaluation with pycocotools — per-class, head/mid/tail, and area-bucket AP. Supports `--weights` and `--out`. |
| `wait_eval_baseline.sh` | Helper: blocks until baseline training finishes, then runs the test eval automatically. |

## Usage

All scripts use a `ROOT` directory for outputs and a `REAL_DATA` path to the CTPD images + labels. **Edit the path constants at the top of each script** (`ROOT`, `REAL_DATA`) to match your local layout before running.

```bash
# 1) Generate 800 synthetic tail samples (needs SDXL weights + GPU)
python motifdiff_run.py --stage gen

# 2) Build the combined (real + synthetic) training manifest + YAML
python motifdiff_run.py --stage yaml

# 3) Train Setting B (real + synthetic)
python motifdiff_run.py --stage train

# 4) Train Setting A (real only) for the controlled baseline
python baseline_train.py

# 5) Evaluate either setting on the official test split
python motifdiff_eval.py --weights runs_motifdiff/motifdiff/weights/best.pt --out motifdiff_eval.json
python motifdiff_eval.py --weights runs_baseline/baseline/weights/best.pt   --out baseline_eval.json
```

## Results (official 862-image test split)

| Setting | mAP@.5 | mAP@.5:.95 | head | mid | tail | AP_small |
|---------|--------|------------|------|-----|------|----------|
| A (real-only)      | 0.461 | 0.346 | 0.522 | 0.550 | 0.076 | 0.246 |
| B (real + 800 synth)| 0.458 | 0.338 | 0.543 | 0.541 | 0.071 | 0.261 |

Synthesis protocol (treated as an *experimental procedure*, not a method claim): SDXL conditioned on heritage-semantic prompts, foreground-mask pseudo-labeling, and geometric augmentation. **No CLIP-based quality filter or curriculum schedule is applied**; these are discussed as future-work / ablation gaps in the paper.

## Data availability

- CTPD dataset: Science Data Bank, DOI [10.57760/sciencedb.34731](https://doi.org/10.57760/sciencedb.34731) (CC BY 4.0).
- 800 synthesized tail-class samples: archived on Zenodo — **DOI to be assigned**.
- This code: https://github.com/oddplant/MotifDiff-CTPD

## License

MIT — see [LICENSE](LICENSE).
