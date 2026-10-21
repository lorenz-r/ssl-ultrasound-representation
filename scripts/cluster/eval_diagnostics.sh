#!/bin/bash
# The diagnostics eval_matrices.sh does not run: RankMe, qualitative, geometry control.
#
# eval_matrices.sh only reran nvidia_matrix.py / stanford_matrix.py. It never
# reran the RankMe collapse diagnostic, the qualitative pass, or the §15
# geometry control for the new encoder — so nobody has checked whether
# vicreg_vit_A collapses on Stanford the way Policy B did (§14.10). Its
# crop-level echogenicity R^2 (0.9716, mm-matched) is the same suspicious
# magnitude as Policy B's collapsed 0.964, which is exactly the false-positive
# pattern RankMe exists to catch. This chain closes that gap, mirroring what
# eval_geometry_control.sh / the WI-17 rankme re-run did for the I-JEPA column.
set -u
REPO=/mnt/home/benbarkow/ssl-ultrasound-representation
LOG=/mnt/home/benbarkow/logs/eval_diagnostics.log
SENTINEL=/mnt/home/benbarkow/logs/eval_diagnostics_sentinel.log

cd "$REPO" || exit 1
source /opt/venvs/.venv/bin/activate

run () {  # run <label> <args...>
  local label="$1"; shift
  echo "[diag] $(date '+%F %T') START $label" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "[diag] $(date '+%F %T') END   $label rc=$rc" >> "$LOG"
  return $rc
}

echo "[diag] === WI-18 diagnostic chain starting $(date '+%F %T') ===" >> "$LOG"

# RankMe collapse diagnostic — auto-discovers vicreg_vit_A, no --encoders needed.
run "rankme mm_matched"     python -u scripts/eval/rankme.py --k 4 --variant mm_matched
run "rankme lambda_matched" python -u scripts/eval/rankme.py --k 4 --variant lambda_matched

# Qualitative pass — DEFAULT_GRID_ENCODERS excludes both vicreg_vit_A and
# ijepa, so both are added explicitly (ijepa was never added here either;
# its qual figures still predate WI-17 training completion).
QUAL_ENCODERS="vicreg_A vicreg_B vicreg_vit_A ijepa mae_raw random_resnet18"
run "qual mm_matched"     python -u scripts/eval/stanford_qualitative.py --k 4 --variant mm_matched \
    --encoders $QUAL_ENCODERS
run "qual lambda_matched" python -u scripts/eval/stanford_qualitative.py --k 4 --variant lambda_matched \
    --encoders $QUAL_ENCODERS

# §15 geometry control — auto-discovers vicreg_vit_A same as nvidia_matrix.py.
for split in validation train; do
  run "geom $split" python -u scripts/eval/geometry_control.py --k 4 --split "$split" \
      --out "outputs/nvidia/geometry_k4_${split}"
  if [ -f "outputs/nvidia/geometry_k4_${split}/results.json" ]; then
    cp "outputs/nvidia/geometry_k4_${split}/results.json" "results/nvidia/geometry_k4_${split}.json"
    echo "[diag] copied geom $split" >> "$LOG"
  fi
done

echo "[diag] === WI-18 diagnostic chain done $(date '+%F %T') ===" >> "$LOG"
echo "=== WI-18 diagnostic chain exited $(date '+%F %T')" >> "$SENTINEL"
