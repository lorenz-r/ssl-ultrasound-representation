#!/bin/bash
# Re-run the NVIDIA + Stanford results matrices over every registered encoder.
#
# No new code path, and deliberately the *same four invocations* as
# `eval_matrices.sh`: `eval/encoders.build_encoders` already registers
# `vicreg_vit_A` the moment `wi18_vicreg_vit_k4_policyA/last.pt` is present
# (it was written gated on exactly that), so the only thing that changes
# between the WI-17 tables and these is one extra column. Anything else that
# differed — a different store, a different crop count — would confound the
# two readings this run exists to supply:
#
#   vicreg_A     -> vicreg_vit_A   isolates the backbone (objective fixed)
#   vicreg_vit_A -> ijepa          isolates the objective (backbone fixed)
#
# NVIDIA first (both splits), then the Stanford transfer table (both grids) —
# §14.10 is the reason the transfer column matters: a representation that
# collapses on real tissue is invisible in the NVIDIA numbers.
set -u
REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
LOG=/mnt/home/benbarkow/logs/eval_matrices.log
SENTINEL=/mnt/home/benbarkow/logs/eval_matrices_sentinel.log

cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

run () {  # run <label> <args...>
  local label="$1"; shift
  echo "[chain] $(date '+%F %T') START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[chain] $(date '+%F %T') END   $label rc=$rc" >> "$LOG"
  return $rc
}

echo "[chain] === eval chain starting $(date '+%F %T') ===" >> "$LOG"

run "nvidia k4 validation" python -u scripts/eval/nvidia_matrix.py --k 4 --split validation \
    --out outputs/nvidia/k4_validation
run "nvidia k4 train"      python -u scripts/eval/nvidia_matrix.py --k 4 --split train \
    --out outputs/nvidia/k4_train
run "stanford mm_matched k4"     python -u scripts/eval/stanford_matrix.py --variant mm_matched --k 4
run "stanford lambda_matched k4" python -u scripts/eval/stanford_matrix.py --variant lambda_matched --k 4

# Refresh the durable copies in results/. The layout mirrors outputs/ exactly
# — outputs/<corpus>/<column>/results.json -> results/<corpus>/<column>.json —
# so a new column needs no new mapping here, only a run line above.
for dir in \
  outputs/nvidia/k4_validation \
  outputs/nvidia/k4_train \
  outputs/stanford/mm_matched_k4 \
  outputs/stanford/lambda_matched_k4 ; do
  src="$dir/results.json"; dst="results/${dir#outputs/}.json"
  if [ -f "$src" ]; then cp "$src" "$dst"; echo "[chain] copied $src -> $dst" >> "$LOG"; fi
done

echo "[chain] === eval chain done $(date '+%F %T') ===" >> "$LOG"
echo "=== eval chain exited $(date '+%F %T')" >> "$SENTINEL"
