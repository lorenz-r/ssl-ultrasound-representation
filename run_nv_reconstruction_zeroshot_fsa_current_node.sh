#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/mnt/home/muhe/ssl-ultrasound-representation}
JOB_TAG=${SLURM_JOB_ID:?Run inside the current interactive Slurm allocation}
SOURCE_DIR=${SOURCE_DIR:-/mnt/home/muhe/data/fsa_mae_20260707}
LOCAL_DIR=${LOCAL_DIR:-/tmp/${USER}/fsa-zeroshot-${JOB_TAG}}
DATA_ZIP=${LOCAL_DIR}/ultrasonic_dataset_fsa.zarr.zip
DATA_METADATA=${LOCAL_DIR}/ultrasonic_dataset_fsa.metadata.json
BEAMFORM_PARAMS=${LOCAL_DIR}/beamform_params_fsa.json
CHECKPOINT=${CHECKPOINT:-/mnt/home/muhe/checkpoints/nv_wide2056_reconstruction_recovered_20260822/best_model_only.pt}
RESULT_ROOT=${RESULT_ROOT:-/mnt/home/muhe/results/nv_reconstruction_zeroshot_rat_fsa}
ENROOT_ROOT=/tmp/${USER}/enroot-${JOB_TAG}
CONTAINER_NAME=ssl-us-${JOB_TAG}
PYTHON=/tmp/${USER}/torch-sm52-${JOB_TAG}/bin/python
GPU_LOCK=/tmp/${USER}/gpu-${JOB_TAG}.lock

mkdir -p "${LOCAL_DIR}" "${RESULT_ROOT}/logs"
for name in ultrasonic_dataset_fsa.metadata.json beamform_params_fsa.json; do
  cp "${SOURCE_DIR}/${name}" "${LOCAL_DIR}/${name}"
done

if [[ ! -f "${DATA_ZIP}" ]] || \
   [[ "$(stat -c %s "${DATA_ZIP}" 2>/dev/null || echo 0)" != \
      "$(stat -c %s "${SOURCE_DIR}/ultrasonic_dataset_fsa.zarr.zip")" ]]; then
  tmp=${DATA_ZIP}.tmp
  trap 'rm -f "${tmp:-}"' EXIT
  echo "Staging 50 GB rat FSA zarr to node-local storage"
  cp "${SOURCE_DIR}/ultrasonic_dataset_fsa.zarr.zip" "${tmp}"
  mv "${tmp}" "${DATA_ZIP}"
  trap - EXIT
fi

export ENROOT_RUNTIME_PATH=${ENROOT_ROOT}/runtime
export ENROOT_CACHE_PATH=${ENROOT_ROOT}/cache
export ENROOT_DATA_PATH=${ENROOT_ROOT}/data

echo "Waiting for the shared Maxwell container environment"
until [[ -x "${PYTHON}" || -L "${PYTHON}" ]] && \
  enroot start --rw -m /tmp:/tmp "${CONTAINER_NAME}" /bin/bash -c \
    "export LD_LIBRARY_PATH=/opt/conda/lib:\${LD_LIBRARY_PATH:-}; \
     ${PYTHON} -c 'import torch, zarr, pymust, wandb'" >/dev/null 2>&1; do
  sleep 20
done

if [[ -f "${HOME}/.wandb_env" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.wandb_env"
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 1
fi

run_adapter() {
  local adapter=$1
  local out_dir=${RESULT_ROOT}/${adapter}
  mkdir -p "${out_dir}"
  enroot start --rw \
    -m "${REPO_ROOT}:/workspace" \
    -m /mnt/home/muhe:/mnt/home/muhe \
    -m /mnt/projects:/mnt/projects \
    -m /tmp:/tmp \
    -e "WANDB_API_KEY=${WANDB_API_KEY}" \
    -e WANDB_MODE=online -e WANDB_REQUIRE_ONLINE=1 \
    "${CONTAINER_NAME}" /bin/bash -c "set -euo pipefail; \
      export LD_LIBRARY_PATH=/opt/conda/lib:\${LD_LIBRARY_PATH:-}; \
      cd /workspace; MPLCONFIGDIR=/tmp/matplotlib-cache \
      PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=.:src \
      ${PYTHON} scripts/evaluate_nv_reconstruction_zeroshot_fsa.py \
        --checkpoint '${CHECKPOINT}' \
        --local-zarr '${DATA_ZIP}' --local-metadata '${DATA_METADATA}' \
        --beamform-params '${BEAMFORM_PARAMS}' --out-dir '${out_dir}' \
        --adapter '${adapter}' --split test --sample-indices 0 1 2 \
        --device cuda --nx 96 --nz 128 --fsa-tx-step 8 --dpi 180 \
        --log-wandb --wandb-project ssl-ultrasound-mae \
        --wandb-group nv-reconstruction-zeroshot-rat-fsa-20260819 \
        --wandb-run-name nv-reconstruction-zeroshot-rat-${adapter}-20260819"
}

# The 12 GB card cannot hold two 2056-wide models. Both experiment pipelines
# run concurrently up to this explicit GPU lock, then take the GPU in turn.
exec 9>"${GPU_LOCK}"
flock 9
run_adapter crop_pad 2>&1 | tee -a "${RESULT_ROOT}/logs/crop_pad.log"
run_adapter resample 2>&1 | tee -a "${RESULT_ROOT}/logs/resample.log"

echo "Rat FSA zero-shot evaluation complete: ${RESULT_ROOT}"
