#!/usr/bin/env bash
set -euo pipefail

# One-command default runner for PRM-as-a-Judge v1.5.
#
# Required (uses the public Preview checkpoint under PRM/ by default):
#   MANIFEST=eval/examples/manifest_demo_cases.jsonl bash eval/run_eval.sh
#
# Useful overrides:
#   GPUS=0,1,2,3 EVAL_MODE=incremental FRAME_INTERVAL=72 BATCH_SIZE=10 bash eval/run_eval.sh
#   PRM=robometer bash eval/run_eval.sh
#   PRM=robometer ROBOMETER_SERVER_URL=http://localhost:8000 bash eval/run_eval.sh
#   VISUALIZE=1 SMOOTHING_WEIGHTS=0.05,0.15,0.6,0.15,0.05 bash eval/run_eval.sh
#   ALLOW_PARTIAL=1 bash eval/run_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST="${MANIFEST:-}"
if [[ -z "${MANIFEST}" ]]; then
  echo "[ERROR] MANIFEST is required. Set it to a JSONL case manifest." >&2
  exit 2
fi
PRM="${PRM:-dopamine}"
USER_PRM_PATH="${PRM_PATH:-${GRM_PATH:-}}"
if [[ "${PRM}" == "robometer" ]]; then
  PRM_PATH="${USER_PRM_PATH:-${PROJECT_ROOT}/../PRM/Robometer-4B}"
else
  PRM_PATH="${USER_PRM_PATH:-${PROJECT_ROOT}/PRM/Robo-Dopamine-GRM-2.0-8B-Preview}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results}"
GPUS="${GPUS:-0}"
EVAL_MODE="${EVAL_MODE:-incremental}"
FRAME_INTERVAL="${FRAME_INTERVAL:-72}"
BATCH_SIZE="${BATCH_SIZE:-10}"
TP_SIZE="${TP_SIZE:-1}"
ROBOMETER_SERVER_URL="${ROBOMETER_SERVER_URL:-}"
ROBOMETER_AUTO_START="${ROBOMETER_AUTO_START:-1}"
ROBOMETER_SERVER_HOST="${ROBOMETER_SERVER_HOST:-127.0.0.1}"
ROBOMETER_SERVER_PORT="${ROBOMETER_SERVER_PORT:-8000}"
ROBOMETER_ROOT="${ROBOMETER_ROOT:-${PROJECT_ROOT}/../robometer}"
if [[ -z "${ROBOMETER_PYTHON:-}" && -x "${PROJECT_ROOT}/../env/robometer/bin/python" ]]; then
  ROBOMETER_PYTHON="${PROJECT_ROOT}/../env/robometer/bin/python"
elif [[ -z "${ROBOMETER_PYTHON:-}" && -x "${PROJECT_ROOT}/../../env/robometer/bin/python" ]]; then
  ROBOMETER_PYTHON="${PROJECT_ROOT}/../../env/robometer/bin/python"
else
  ROBOMETER_PYTHON="${ROBOMETER_PYTHON:-$(command -v python)}"
fi
ROBOMETER_NUM_GPUS="${ROBOMETER_NUM_GPUS:-1}"
ROBOMETER_MAX_WORKERS="${ROBOMETER_MAX_WORKERS:-1}"
ROBOMETER_FRAME_STEPS_MICRO_BATCH_SIZE="${ROBOMETER_FRAME_STEPS_MICRO_BATCH_SIZE:-16}"
ROBOMETER_STARTUP_TIMEOUT_S="${ROBOMETER_STARTUP_TIMEOUT_S:-720}"
ROBOMETER_FPS="${ROBOMETER_FPS:-1.0}"
ROBOMETER_VIEW="${ROBOMETER_VIEW:-video}"
ROBOMETER_TIMEOUT_S="${ROBOMETER_TIMEOUT_S:-1600}"
ROBOMETER_USE_FRAME_STEPS="${ROBOMETER_USE_FRAME_STEPS:-1}"
OUTLIER_METHOD="${OUTLIER_METHOD:-local_median}"
SMOOTHING="${SMOOTHING:-weighted_window}"
SMOOTHING_WEIGHTS="${SMOOTHING_WEIGHTS:-0.1,0.2,0.4,0.2,0.1}"
VISUALIZE="${VISUALIZE:-0}"
VISUALIZE_MAX_CASES="${VISUALIZE_MAX_CASES:-0}"
VISUALIZE_OUTPUT_DIR="${VISUALIZE_OUTPUT_DIR:-}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUNNER_PYTHON="${PYTHON:-python}"
if [[ "${PRM}" == "robometer" && -z "${PYTHON:-}" ]]; then
  RUNNER_PYTHON="${ROBOMETER_PYTHON}"
fi

CMD=("${RUNNER_PYTHON}" "${SCRIPT_DIR}/run_judge.py" eval
  --manifest "${MANIFEST}"
  --prm "${PRM}"
  --prm-path "${PRM_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --gpus "${GPUS}"
  --eval-mode "${EVAL_MODE}"
  --frame-interval "${FRAME_INTERVAL}"
  --batch-size "${BATCH_SIZE}"
  --tensor-parallel-size "${TP_SIZE}"
  --outlier-method "${OUTLIER_METHOD}"
  --smoothing "${SMOOTHING}"
  --smoothing-weights "${SMOOTHING_WEIGHTS}"
)

if [[ "${PRM}" == "robometer" ]]; then
  CMD+=(
    --robometer-server-host "${ROBOMETER_SERVER_HOST}"
    --robometer-server-port "${ROBOMETER_SERVER_PORT}"
    --robometer-root "${ROBOMETER_ROOT}"
    --robometer-python "${ROBOMETER_PYTHON}"
    --robometer-num-gpus "${ROBOMETER_NUM_GPUS}"
    --robometer-max-workers "${ROBOMETER_MAX_WORKERS}"
    --robometer-frame-steps-micro-batch-size "${ROBOMETER_FRAME_STEPS_MICRO_BATCH_SIZE}"
    --robometer-startup-timeout-s "${ROBOMETER_STARTUP_TIMEOUT_S}"
    --robometer-fps "${ROBOMETER_FPS}"
    --robometer-view "${ROBOMETER_VIEW}"
    --robometer-timeout-s "${ROBOMETER_TIMEOUT_S}"
  )
  if [[ -n "${ROBOMETER_SERVER_URL}" ]]; then
    CMD+=(--robometer-server-url "${ROBOMETER_SERVER_URL}")
  fi
  if [[ "${ROBOMETER_AUTO_START}" == "0" || "${ROBOMETER_AUTO_START}" == "false" ]]; then
    CMD+=(--no-robometer-auto-start)
  else
    CMD+=(--robometer-auto-start)
  fi
  if [[ "${ROBOMETER_USE_FRAME_STEPS}" == "0" || "${ROBOMETER_USE_FRAME_STEPS}" == "false" ]]; then
    CMD+=(--no-robometer-use-frame-steps)
  else
    CMD+=(--robometer-use-frame-steps)
  fi
fi

if [[ "${VISUALIZE}" == "1" || "${VISUALIZE}" == "true" ]]; then
  CMD+=(--visualize --visualize-max-cases "${VISUALIZE_MAX_CASES}")
  if [[ -n "${VISUALIZE_OUTPUT_DIR}" ]]; then
    CMD+=(--visualize-output-dir "${VISUALIZE_OUTPUT_DIR}")
  fi
fi

if [[ "${ALLOW_PARTIAL}" == "1" || "${ALLOW_PARTIAL}" == "true" ]]; then
  CMD+=(--allow-partial)
fi

echo "[INFO] PRM-as-a-Judge v1.5"
echo "[INFO] prm=${PRM} eval_mode=${EVAL_MODE} frame_interval=${FRAME_INTERVAL} gpus=${GPUS}"
echo "[INFO] smoothing=${SMOOTHING} smoothing_weights=${SMOOTHING_WEIGHTS} visualize=${VISUALIZE}"
if [[ "${PRM}" == "robometer" ]]; then
  echo "[INFO] robometer_server_url=${ROBOMETER_SERVER_URL:-auto:${ROBOMETER_SERVER_HOST}:${ROBOMETER_SERVER_PORT}} auto_start=${ROBOMETER_AUTO_START} fps=${ROBOMETER_FPS} view=${ROBOMETER_VIEW} use_frame_steps=${ROBOMETER_USE_FRAME_STEPS}"
fi
"${CMD[@]}" ${EXTRA_ARGS}
