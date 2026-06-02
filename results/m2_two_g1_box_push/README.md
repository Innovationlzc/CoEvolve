# M2 Two-G1 Box Push Lite

This is an Isaac Lab eval result for `CoEvolve-TwoG1BoxPush-Direct-v0`.

Current scope:

- two visible Unitree G1 bodies;
- one rigid push box;
- low-dimensional observation;
- proxy 4D action where each agent contributes a planar push and the box
  receives the averaged velocity;
- G1 bodies are kept upright as visual actors while the proxy action is still in
  use;
- the robots do not currently push through foot motion, walking, arm motion, or
  contact. The box is moved by the proxy action, and the two G1 visuals follow
  smoothly with fixed joints;
- `--action-mode base_command` is a newer intermediate adapter where actions
  move the G1 base visuals first, then the box moves through a virtual contact
  zone;
- progress metrics written to `metrics.json` and `history.csv`.

Run headless:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py --num_envs 1 --max_steps 600 --viz none
```

Run with Newton visualization:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py --num_envs 1 --max_steps 600 --viz newton
```

Run the trained HALyPO proxy policy in Isaac:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 130 \
  --policy-path results/m2_proxy_training/halypo_policy.json \
  --stop-on-success \
  --viz newton
```

Current result: first success at step 126 with reset-before-distance `0.1878`.
This validates the M2 training/eval plumbing, not real G1 locomotion or contact
control. Next step: replace the proxy box velocity action with a G1 command
adapter.

See `results/m2_base_command_smoke/` for the first base-command adapter smoke:
success at step 104 with robot x behind box x by about `0.7565`.
