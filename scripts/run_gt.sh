Terminal 1:


```bash
cd /data1/wuhaochen/AAAI/CoEvolve/
OUT=results/deployment/payload_5kg_live
mkdir -p "$OUT"
rm -f "$OUT/planner_heartbeat.json" "$OUT/runtime_state.json"

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_dual_sonic_manual_layout_sim.py \
  --interface auto \
  --with-box \
  --box-mass-kg 3.0 \
  --payload-hold \
  --payload-release-after-drop-s 5 \
  --upper-body-lock-pose elbow110_carry_ready \
  --upper-body-lock-kp 24 \
  --upper-body-lock-kd 1.5 \
  --upper-body-lock-ramp-s 6 \
  --upper-body-lock-start-after-drop-s 1 \
  --planner-heartbeat-drop-phase-keyword static_skill_pre_stand \
  --auto-drop-when-ready \
  --lowcmd-warmup-s 4 \
  --min-drop-time-s 4 \
  --planner-heartbeat-file "$OUT/planner_heartbeat.json" \
  --planner-heartbeat-max-age-s 1
```

Terminal 2:

```bash
cd /data1/wuhaochen/AAAI/CoEvolve/
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id front --dds-topic-prefix front --zmq-port 5556 --zmq-out-port 5557
```

Terminal 3:

```bash
cd /data1/wuhaochen/AAAI/CoEvolve/
scripts/run_official_sonic_deploy_zmq_manager.sh auto \
  --robot-id back --dds-topic-prefix back --zmq-port 5558 --zmq-out-port 5559
```

Terminal 4

```bash
# cd /data1/wuhaochen/AAAI/CoEvolve/

# OUT=/data1/wuhaochen/AAAI/CoEvolve/results/deployment/payload_5kg_live

# PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
#   scripts/run_sonic_static_skill.py \
#   --skill carry_forward \
#   --distance-m 0 \
#   --speed-mps 0 \
#   --start-pulse-seconds 0 \
#   --stand-seconds 0 \
#   --final-idle-seconds 2 \
#   --no-stop \
#   --heartbeat-file "$OUT/planner_heartbeat.json"

# PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
#   scripts/run_sonic_static_skill.py \
#   --skill carry_forward \
#   --distance-m 15 \
#   --speed-mps 0.3 \
#   --start-pulse-seconds 0 \
#   --stand-seconds 0 \
#   --final-idle-seconds 2 \
#   --no-stop \
#   --heartbeat-file "$OUT/planner_heartbeat.json"

cd /data1/wuhaochen/AAAI/CoEvolve/

OUT=/data1/wuhaochen/AAAI/CoEvolve/results/deployment/payload_5kg_live

## 初始化接住箱子
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --distance-m 3 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"

## 向前走到转弯处
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --distance-m 12 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"

## 开始转弯
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --robots front \
  --distance-m 3 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"

  PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_turn_in_place \
  --robots front back \
  --yaw-deg 45 \
  --yaw-rate-degps 3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"


PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --robots front \
  --distance-m 2 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json" 

  PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_turn_in_place \
  --robots front back \
  --yaw-deg 45 \
  --yaw-rate-degps 3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"

PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --robots front \
  --distance-m 2 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json" 


## 只走通过长廊
PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/run_sonic_static_skill.py \
  --skill carry_forward \
  --distance-m 12 \
  --speed-mps 0.3 \
  --start-pulse-seconds 0 \
  --stand-seconds 0 \
  --final-idle-seconds 2 \
  --no-stop \
  --heartbeat-file "$OUT/planner_heartbeat.json"

```

