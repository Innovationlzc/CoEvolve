#!/usr/bin/env bash
set -eo pipefail

# Resolve this checkout rather than binding deployment to one workstation.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
INTERFACE="${SONIC_DDS_INTERFACE:-auto}"
ROBOT_ID="${SONIC_ROBOT_ID:-g1}"
DDS_TOPIC_PREFIX="${SONIC_DDS_TOPIC_PREFIX:-}"
ZMQ_PORT="${SONIC_ZMQ_PORT:-5556}"
ZMQ_OUT_PORT="${SONIC_ZMQ_OUT_PORT:-5557}"
ZMQ_OUT_TOPIC="${SONIC_ZMQ_OUT_TOPIC:-g1_debug}"
LOG_DIR="${SONIC_DEPLOY_LOG_DIR:-$REPO_ROOT/results/deployment/official_sonic_motor_preflight/deploy_logs}"
FORCE_REBUILD="${SONIC_FORCE_REBUILD:-0}"

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
    --rebuild)
      FORCE_REBUILD=1
      shift
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

DEPLOY_SOURCE="$DEPLOY_DIR/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp"
if [[ -n "$DDS_TOPIC_PREFIX" ]] && ! grep -q -- "--dds-topic-prefix" "$DEPLOY_SOURCE"; then
  echo "[praxis-sonic-deploy] ERROR: gear_sonic_deploy does not support --dds-topic-prefix." >&2
  echo "[praxis-sonic-deploy] Apply and rebuild the CoEvolve SONIC patches first:" >&2
  echo "  cd $REPO_ROOT/third_party/GR00T-WholeBodyControl" >&2
  echo "  git apply $REPO_ROOT/patches/gear_sonic_multi_robot_dds_prefix.patch" >&2
  echo "  git apply $REPO_ROOT/patches/gear_sonic_heading_increment.patch" >&2
  echo "  cd gear_sonic_deploy && source scripts/setup_env.sh && just --justfile .justfile build" >&2
  exit 1
fi

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
BUILD_LOCK="$DEPLOY_DIR/.coevolve_build.lock"
DEPLOY_BINARY="$DEPLOY_DIR/target/release/g1_deploy_onnx_ref"
NEEDS_BUILD=0
if [[ "$FORCE_REBUILD" == "1" || ! -x "$DEPLOY_BINARY" ]]; then
  NEEDS_BUILD=1
elif find "$DEPLOY_DIR/src" "$DEPLOY_DIR/CMakeLists.txt" "$DEPLOY_DIR/.justfile" \
  -type f -newer "$DEPLOY_BINARY" -print -quit | grep -q .; then
  NEEDS_BUILD=1
fi

if [[ "$NEEDS_BUILD" == "1" ]]; then
  (
    flock 9
    just --justfile .justfile build
  ) 9>"$BUILD_LOCK"
else
  echo "[praxis-sonic-deploy] reusing current deploy binary (pass --rebuild after source changes)"
fi

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
