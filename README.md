# MotifDiff-CTPD

Code and artifacts accompanying **"MotifDiff: A Controlled Evaluation of Generative Augmentation for Traditional Chinese Pattern Detection"** (paper 6, submitted to *Neurocomputing*).

## What is here

| File | Purpose |
|------|---------|
| `paper6_extra_exp.py` | The A/B/C/D evaluation protocol: YOLOv8-s training on CTPD + pycocotools-based grouped / area-bucket / per-class evaluation. |
| `fid_kid.py` | FID / KID computation (InceptionV3 pool3 features) for §9.3 quantitative domain-shift analysis. |
| `motifdiff_eval.py` | Evaluation utilities shared by the protocol. |
| `motifdiff_run.py` | SDXL synthesis driver (layout-controllable, heritage-semantic prompted). |
| `data/*.yaml` | CTPD dataset split definitions. `path:` points to a local CTPD checkout; download CTPD from Science Data Bank (see Data below). |
| `prompts/*.json` | Heritage-semantic prompt tables used for synthesis and OVD prompting. |

## Data

- **CTPD dataset** (real images + annotations): Science Data Bank, DOI [10.57760/sciencedb.34731](https://doi.org/10.57760/sciencedb.34731), CC BY 4.0.
- **1,800 synthesized samples** (800 tail-class + 1,000 mid-frequency): archived on Zenodo, DOI: _to be assigned_.

## Reproduce

```bash
# 1. train + evaluate the four settings (A real-only, B +800 tail, C +400 CLIP-filtered tail, D +1000 mid-frequency)
python paper6_extra_exp.py --stage train_a   # ... train_b / train_c / train_d, then evaluate

# 2. quantitative domain shift (§9.3)
python fid_kid.py   # needs real CTPD images (pattern/data/images) + synthesized samples
```

## License

Code and prompts: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
Synthesized samples: CC BY 4.0 (see Zenodo archive).
