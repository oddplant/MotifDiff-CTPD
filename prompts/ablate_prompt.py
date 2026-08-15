# ablate_prompt.py — prompt-length ablation for OVD zero-shot on CTPD TEST split.
# Reuses eval_ovd.evaluate() so the mAP pipeline is identical to the main results.
#   SHORT = bare class names (shortest)      -> eval_ovd.NAMES_BARE
#   MED   = concise cultural phrase (middle) -> defined below
#   LONG  = full cultural description (long) -> eval_ovd.NAMES_SEM
# Goal: show detection quality does NOT improve with longer prompts (refutes
#       "more words is better"; HSP is a diagnostic probe, not a length trick).
import os, json, sys, time
sys.path.insert(0, "/LiKun/crop-detection/pattern")
import eval_ovd as E
import torch
torch.backends.cudnn.enabled = False
from ultralytics import YOLOWorld

WORLD_W = "/LiKun/crop-detection/pattern/yolov8l-world.pt"

# MED: one concise cultural phrase per class, aligned to NAMES_BARE order (0..16)
NAMES_MED = [
    "the Eight Buddhist auspicious symbols as a combined emblem",            # 0 bajixiangwen
    "an auspicious rosette medallion flower of Tang-Song court art",         # 1 baoxianghuawen
    "the Chinese character fu meaning good fortune",                         # 2 fuziwen
    "a gong-shaped silver ingot wealth motif",                               # 3 gongziwen
    "a cowrie-shell coin wealth motif of the pre-Qin era",                   # 4 gubeiwen
    "a framed barrel drum musical ornament",                                 # 5 guwen
    "a crane bird, a Taoist symbol of longevity",                            # 6 hewen
    "a tiger beast motif for folk protection",                               # 7 huwen
    "a slender banana-leaf border ornament on bronzes",                      # 8 jiaoyewen
    "a lotus bloom, the Buddhist sacred flower",                             # 9 lianhuawen
    "an endless interlaced knot of the Eight symbols",                       # 10 panchangwen
    "the Chinese character shou meaning longevity",                          # 11 shouziwen
    "a sun disc with radiating rays",                                        # 12 taiyangwen
    "a right-turning swastika auspicious fret border",                       # 13 wanziwen
    "the Chinese character xi meaning joy",                                  # 14 xiziwen
    "a pair of mandarin ducks, a love symbol",                               # 15 yuanyangwen
    "a fish, homophone for surplus and abundance",                           # 16 yuwen
]

def run(tag, NAMES):
    out = E.evaluate(tag, WORLD_W, NAMES=NAMES, is_ovd=True)
    p = "/LiKun/crop-detection/pattern/eval_prompt_%s.json" % tag
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print("=== %s === mAP50=%.4f mAP50_95=%.4f head=%.4f mid=%.4f tail=%.4f small=%.4f" % (
        tag, out["mAP50"], out["mAP50_95"], out["head_ap"], out["mid_ap"],
        out["tail_ap"], out["ap_small_lt1"]), flush=True)
    return out

if __name__ == "__main__":
    t0 = time.time()
    # SHORT=bare, LONG=full semantic (both already in eval_ovd); MED=new middle tier
    for tag, NAMES in [("short", E.NAMES_BARE), ("med", NAMES_MED), ("long", E.NAMES_SEM)]:
        run(tag, NAMES)
    print("ALL DONE in %.1fs" % (time.time() - t0), flush=True)
