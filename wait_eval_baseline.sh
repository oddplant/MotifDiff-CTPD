#!/bin/bash
PID=75064
echo "[wait] waiting for baseline train PID $PID to finish..."
while kill -0 $PID 2>/dev/null; do sleep 15; done
echo "[wait] baseline train done, starting test eval..."
cd /LiKun/crop-detection/paper6_MotifDiff
PY=/mnt/inspurfs/user-fs/080221/anaconda/envs/pytorch/bin/python
$PY motifdiff_eval.py --weights runs_baseline/baseline/weights/best.pt --out baseline_eval.json
echo "[wait] BASELINE TEST EVAL DONE"
