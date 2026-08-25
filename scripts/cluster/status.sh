#!/bin/bash
# Compact health summary for the overnight VICReg runs. Read-only.
LOG_DIR=/mnt/home/benbarkow/logs
CKPT_ROOT=/mnt/home/benbarkow/checkpoints

echo "=== $(date '+%F %T') ==="
echo "--- chain ---"
tail -6 "$LOG_DIR/train_vicreg_policies.log" 2>/dev/null

echo "--- processes / GPU ---"
echo "train procs: $(pgrep -cf 'train/vicreg.py --policy')"
nvidia-smi 2>/dev/null | sed -n '10p'
echo "cgroup mem: $(awk 'BEGIN{}/^anon /{a=$2}/^file /{f=$2}END{printf "anon %.1f GB, file %.1f GB", a/1e9, f/1e9}' /sys/fs/cgroup/memory.stat) / 24.0 GB"

for P in A B; do
  LOG="$LOG_DIR/vicreg_policy${P}.log"
  [ -f "$LOG" ] || continue
  echo "--- policy $P ---"
  grep -E "Traceback|Error|Killed|CUDA out of memory|HALT" "$LOG" | tail -3
  tail -3 "$LOG" | grep -E "^\[e|^\[wi12\]|^\[resume\]"
  python3 - "$CKPT_ROOT/wi12_vicreg_policy${P}" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
try:
    h = json.load(open(d / "history.json"))
except Exception:
    sys.exit()
n = len(h.get("loss", []))
if not n:
    sys.exit()
# The epoch target follows from the pair budget and the measured steps/epoch,
# so read it back rather than hardcoding a number that drifts when either moves.
try:
    total = int(json.load(open(d / "run_config.json"))["epochs"])
except Exception:
    total = 223
def fmt(k, p=2):
    v = h.get(k, [])
    if not v: return "n/a"
    return f"{v[0]:.{p}f} -> {v[-1]:.{p}f}"
secs = h.get("epoch_seconds", [])
recent = secs[-10:] or [0]
avg = sum(recent)/len(recent)
left = (total - n) * avg / 3600
print(f"  epoch {n}/{total} | loss {fmt('loss')} | RankMe {fmt('rankme',1)}")
print(f"  inv {fmt('invariance',3)} | var {fmt('variance',3)} | cov {fmt('covariance',3)}")
pps = h.get("pairs_per_second", [])
if pps:
    print(f"  {avg:.0f}s/epoch, {pps[-1]:.0f} pairs/s, ETA {left:.1f} h")
r = h.get("rankme", [])
if len(r) >= 6:
    print(f"  RankMe last 6: {[round(x,1) for x in r[-6:]]}")
PY
done
