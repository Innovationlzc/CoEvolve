# M3 OSP-Alignment-Lite

First paper-bridge Isaac task for HALyPO reproduction.

Implemented now:

- OSP-style directional alignment scene with a 20 kg long object and narrow gate.
- One Unitree G1 visual/controller side plus a scripted human/proxy partner side.
- 11D MARL command surface: base vx/vy/yaw, CoM height, torso pitch, and left/right wrist XYZ deltas.
- G1 wrist target markers and human/proxy wrist markers.
- Distance-based hand/object contact scores and reward penalties for contact loss.
- 210D policy observation shape via three 70D paper-inspired temporal snapshots.
- Temporary root-stabilized arm tracker for smoke testing.

Important limitation: this is not yet true G1 locomotion, WBC, or physical hand-object
manipulation. Object motion is still produced by a contact-gated velocity/torque
adapter so the controller dependency stays visible.

Headless smoke:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m3_osp_alignment.py \
  --num_envs 2 \
  --max_steps 180 \
  --log-every 20 \
  --viz none \
  --output-dir results/m3_osp_alignment_lite
```

Newton visualization:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m3_osp_alignment.py \
  --num_envs 1 \
  --max_steps 180 \
  --log-every 20 \
  --viz newton \
  --output-dir results/m3_osp_alignment_lite
```

Next fidelity step: replace the temporary tracker with the Isaac Lab G1 Pink IK
upper-body action and/or a G1 WBC command adapter.

