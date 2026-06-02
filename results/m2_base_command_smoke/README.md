# M2 Base-Command Adapter Smoke

This result uses `--action-mode base_command`.

Current behavior:

- the 4D action first updates the two G1 base visual positions;
- the box is pushed only through a simple virtual contact zone behind the box;
- G1 joints/feet are still fixed, so this is not walking, WBC, or true rigid-body
  contact pushing yet;
- unlike `proxy_box_velocity`, the G1 visuals no longer just follow the box. They
  move forward under the command and remain behind the box.

Current result:

- first success step: `104`;
- best distance: `0.1921`;
- final box x: `1.8079`;
- final robot x: `1.0514`;
- robot-box standoff: about `0.7565`.

Run:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 120 \
  --log-every 20 \
  --action-mode base_command \
  --stop-on-success \
  --viz newton \
  --output-dir results/m2_base_command_smoke
```
