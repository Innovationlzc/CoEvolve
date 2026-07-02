# CoEvolve / Praxis-Coop

Praxis-Coop explores verified cooperative skill orchestration above a frozen
humanoid whole-body controller.

Current direction:

> Official SONIC / GEAR-SONIC motor execution + cooperative SkillGraph
> orchestration + verifier-gated deployment SpecMemory.

The old HALyPO / MARL reproduction work is local archive material only. The
current repository is organized for the new SONIC-based project.

## Current Status

The current reproducible milestone is a motor-layer baseline:

- generate a dual-G1 MuJoCo scene from the user-authored manual USD layout;
- run two official `gear_sonic_deploy` processes, one per G1;
- send synchronized ZMQ planner commands;
- use lowcmd readiness plus planner heartbeat before dropping the virtual
  support band;
- visually pass dual-G1 no-box synchronized stand/walk.

This is not yet a physical box-transport result.

## Repository Layout

| Path | Purpose |
|---|---|
| `assets/manual_scenes/two_g1_box_push_layout.usda` | tracked user-authored manual scene |
| `src/praxis_coop/` | current implementation package |
| `scripts/run_dual_sonic_manual_layout_sim.py` | dual-G1 MuJoCo runtime wrapper |
| `scripts/run_official_sonic_deploy_zmq_manager.sh` | official GEAR-SONIC deploy launcher |
| `scripts/run_dual_official_sonic_motor_preflight.py` | synchronized planner command publisher |
| `docs/README.md` | documentation index |
| `docs/REPRODUCTION.md` | detailed current run commands |
| `docs/SYSTEM_DESIGN.md` | method architecture |
| `docs/SONIC_INTEGRATION.md` | SONIC adapter boundary |
| `results/spec_memory/static/` | curated reusable deployment specs |

Local-only paths ignored by git:

| Path | Reason |
|---|---|
| `third_party/GR00T-WholeBodyControl/` | external SONIC runtime |
| `models/` | Qwen checkpoints |
| `results/deployment/` | generated experiment artifacts |
| `HaLyPO/` | old archived work, not current evidence |
| `source/*.pdf` | local paper PDFs |

## Environment Setup

The commands below assume Ubuntu and a clone at `/home/admin1/Project/CoEvolve`.
If you clone elsewhere, replace that path consistently.

### 1. Clone This Repository

```bash
mkdir -p /home/admin1/Project
cd /home/admin1/Project
git clone git@github.com:Innovationlzc/CoEvolve.git
cd CoEvolve
```

### 2. Install Basic System Tools

```bash
sudo apt update
sudo apt install -y git git-lfs curl build-essential cmake clang pv
git lfs install
```

Install `just` locally if it is not already available:

```bash
mkdir -p ~/.local/bin
curl -L https://github.com/casey/just/releases/download/1.43.0/just-1.43.0-x86_64-unknown-linux-musl.tar.gz \
  | tar xz -C /tmp
install -m 755 /tmp/just ~/.local/bin/just
export PATH="$HOME/.local/bin:$PATH"
just --version
```

### 3. Prepare Isaac Lab Runtime

Isaac Lab is treated as an external runtime, not vendored in this repository.
Current local commands assume:

```text
/home/admin1/Project/IsaacLab/env_isaaclab
```

Activate it when running tests or Isaac-dependent utilities:

```bash
cd /home/admin1/Project/IsaacLab
source env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES
```

### 4. Install Local Python Package Helpers

For non-Isaac Python utilities:

```bash
cd /home/admin1/Project/CoEvolve
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e . pytest
```

The main validation command below uses Isaac Lab's Python because the local
machine already keeps Isaac/PXR-related packages there.

### 5. Clone And Set Up SONIC / GR00T-WholeBodyControl

```bash
cd /home/admin1/Project/CoEvolve
mkdir -p third_party
if [ ! -d third_party/GR00T-WholeBodyControl/.git ]; then
  git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git third_party/GR00T-WholeBodyControl
fi

cd third_party/GR00T-WholeBodyControl
git lfs pull
bash install_scripts/install_mujoco_sim.sh
```

Install the SONIC Python package into the MuJoCo virtual environment:

```bash
cd /home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl
.venv_sim/bin/python -m pip install -U pip huggingface_hub
.venv_sim/bin/python -m pip install -e .
```

Download the official GEAR-SONIC deployment models:

```bash
cd /home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl
.venv_sim/bin/python download_from_hf.py
```

This should create files such as:

```text
gear_sonic_deploy/policy/release/model_encoder.onnx
gear_sonic_deploy/policy/release/model_decoder.onnx
gear_sonic_deploy/policy/release/observation_config.yaml
gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

### 6. Prepare TensorRT And Deploy Build

The official C++ deploy stack expects TensorRT. On the current workstation we
use TensorRT under `~/TensorRT`.

After downloading the TensorRT 10.13 Linux tarball from NVIDIA, extract it:

```bash
cd ~/Downloads
tarball=$(ls TensorRT-10.13*.tar.gz | head -1)
pv "$tarball" | tar -xz -f -
trt_dir=$(tar -tf "$tarball" | head -1 | cut -d/ -f1)
rm -rf ~/TensorRT
mv "$trt_dir" ~/TensorRT

grep -qxF 'export TensorRT_ROOT=$HOME/TensorRT' ~/.bashrc || \
  echo 'export TensorRT_ROOT=$HOME/TensorRT' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$HOME/TensorRT/lib:$LD_LIBRARY_PATH' ~/.bashrc || \
  echo 'export LD_LIBRARY_PATH=$HOME/TensorRT/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

export TensorRT_ROOT=$HOME/TensorRT
export LD_LIBRARY_PATH=$HOME/TensorRT/lib:$LD_LIBRARY_PATH
```

Build the deploy target:

```bash
cd /home/admin1/Project/CoEvolve/third_party/GR00T-WholeBodyControl/gear_sonic_deploy
chmod +x scripts/install_deps.sh
./scripts/install_deps.sh

export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export TensorRT_ROOT="${TensorRT_ROOT:-$HOME/TensorRT}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-}"
export onnxruntime_DIR="${onnxruntime_DIR:-}"

source scripts/setup_env.sh
just --justfile .justfile build
```

The CoEvolve wrapper also runs the deploy build before launching:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

## Validation

Run unit tests:

```bash
cd /home/admin1/Project/CoEvolve
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python -m pytest -q
```

Expected current result:

```text
21 passed
```

Generate and compile-check the dual-G1 no-payload MuJoCo scene:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/build_sonic_manual_layout_scene.py \
  --dual-g1 \
  --no-payload \
  --compile-check
```

Expected tail:

```text
compile_check=ok
```

## Known-Good Run: Dual-G1 No-Box Synchronized Walk

This is the current stable baseline that has visually passed. Use four
terminals. Keep the heartbeat path identical in Terminal 1 and Terminal 4.

Terminal 1, dual-G1 MuJoCo runtime:

```bash
cd /home/admin1/Project/CoEvolve
rm -f results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --upper-body-lock-pose none \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 2 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json \
  --planner-heartbeat-max-age-s 1
```

Terminal 2, front G1 deploy:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3, back G1 deploy:

```bash
cd /home/admin1/Project/CoEvolve
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id back --dds-topic-prefix back --zmq-port 5558 --zmq-out-port 5559
```

Terminal 4, after Terminal 2 and 3 print `Init Done`:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_official_sonic_motor_preflight.py \
  --front-port 5556 --back-port 5558 \
  --start-pulse-seconds 8 \
  --stand-seconds 10 \
  --walk-meters 1.0 \
  --walk-speed 0.10 \
  --walk-mode slow_walk \
  --movement-x 1 --movement-y 0 \
  --facing-x 1 --facing-y 0 \
  --upper-body-pose none \
  --hand-pose none \
  --final-idle-seconds 3600 \
  --no-stop \
  --heartbeat-file results/deployment/dual_sonic_selftest_guarded_walk/planner_heartbeat.json \
  --output-dir results/deployment/dual_sonic_selftest_guarded_walk
```

Expected behavior:

- Terminal 1 auto-drops only after both deploys are ready and heartbeat is fresh.
- Both robots stand, then walk forward together under official SONIC control.
- This run has no payload contact and is motor-layer evidence only.

## Optional: Qwen Models

Qwen is not required for the current motor-layer baseline. For later
SkillGraph/Qwen evolution experiments:

```bash
cd /home/admin1/Project/CoEvolve
PYTHONPATH=src /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/download_phase1_models.py --model all
```

Expected ignored local paths:

```text
models/Qwen3-8B
models/Qwen3-VL-8B-Instruct
```

## Boundary

The current valid motor evidence is official SONIC/GEAR-SONIC in MuJoCo. Older
Isaac proxy/root-teleport runs, HTML-only replays, and payload-contact attempts
are not current paper evidence.
