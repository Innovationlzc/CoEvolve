#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="/home/admin1/Project/CoEvolve"
DEPLOY_DIR="$REPO_ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
INTERFACE="${SONIC_DDS_INTERFACE:-auto}"
ROBOT_ID="${SONIC_ROBOT_ID:-g1}"
DDS_TOPIC_PREFIX="${SONIC_DDS_TOPIC_PREFIX:-}"
ZMQ_PORT="${SONIC_ZMQ_PORT:-5556}"
ZMQ_OUT_PORT="${SONIC_ZMQ_OUT_PORT:-5557}"
ZMQ_OUT_TOPIC="${SONIC_ZMQ_OUT_TOPIC:-g1_debug}"
LOG_DIR="${SONIC_DEPLOY_LOG_DIR:-$REPO_ROOT/results/deployment/official_sonic_motor_preflight/deploy_logs}"

if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  INTERFACE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-id)
      ROBOT_ID="$2"
      shift 2
      ;;
    --dds-topic-prefix)
      DDS_TOPIC_PREFIX="$2"
      shift 2
      ;;
    --zmq-port)
      ZMQ_PORT="$2"
      shift 2
      ;;
    --zmq-out-port)
      ZMQ_OUT_PORT="$2"
      shift 2
      ;;
    --zmq-out-topic)
      ZMQ_OUT_TOPIC="$2"
      shift 2
      ;;
    --logs-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$DDS_TOPIC_PREFIX" ]]; then
  LOG_DIR="${LOG_DIR%/}/$ROBOT_ID"
fi

cd "$DEPLOY_DIR"
mkdir -p "$LOG_DIR"

if [[ "$INTERFACE" == "auto" ]]; then
  INTERFACE=""
  while read -r iface; do
    case "$iface" in
      lo|tailscale*|docker*|br-*|veth*|Meta) continue ;;
    esac
    if ip -o link show "$iface" 2>/dev/null | grep -q "BROADCAST.*MULTICAST"; then
      INTERFACE="$iface"
      break
    fi
  done < <(ip -o -4 addr show up scope global 2>/dev/null | awk '{print $2}')
  INTERFACE="${INTERFACE:-lo}"
fi

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export TensorRT_ROOT="${TensorRT_ROOT:-}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-}"
export onnxruntime_DIR="${onnxruntime_DIR:-}"

source scripts/setup_env.sh
just --justfile .justfile build

echo "[praxis-sonic-deploy] using DDS interface: $INTERFACE"
echo "[praxis-sonic-deploy] robot_id: $ROBOT_ID"
echo "[praxis-sonic-deploy] dds_topic_prefix: ${DDS_TOPIC_PREFIX:-<none>}"
echo "[praxis-sonic-deploy] zmq_port: $ZMQ_PORT"
echo "[praxis-sonic-deploy] zmq_out_port: $ZMQ_OUT_PORT"

ZMQ_VERBOSE_ARGS=()
if [[ "${SONIC_ZMQ_VERBOSE:-0}" == "1" ]]; then
  ZMQ_VERBOSE_ARGS+=(--zmq-verbose)
fi
DDS_TOPIC_ARGS=()
if [[ -n "$DDS_TOPIC_PREFIX" ]]; then
  DDS_TOPIC_ARGS+=(--dds-topic-prefix "$DDS_TOPIC_PREFIX")
fi

exec just --justfile .justfile run g1_deploy_onnx_ref "$INTERFACE" policy/release/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host localhost \
  --zmq-port "$ZMQ_PORT" \
  --zmq-out-port "$ZMQ_OUT_PORT" \
  --zmq-out-topic "$ZMQ_OUT_TOPIC" \
  "${ZMQ_VERBOSE_ARGS[@]}" \
  --logs-dir "$LOG_DIR" \
  "${DDS_TOPIC_ARGS[@]}" \
  --disable-crc-check
