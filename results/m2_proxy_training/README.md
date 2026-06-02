# M2 Proxy Training

This trains the current M2 4D proxy action:

```text
[agent_a_push_x, agent_a_push_y, agent_b_push_x, agent_b_push_y]
```

It compares independent updates against HALyPO on a differentiable proxy of the
Isaac box-push task, then writes `halypo_policy.json` for Isaac evaluation.

Current result:

- independent final distance: `0.3370`;
- independent constraint violation rate: `0.0625`;
- HALyPO final distance in proxy: `0.3981`;
- HALyPO constraint violation rate: `0.0`;
- HALyPO Isaac eval success: step 126, best distance `0.1878`.

Run:

```bash
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m2_proxy_train.py
```

Evaluate:

```bash
OMNI_KIT_ACCEPT_EULA=YES /home/admin1/Project/IsaacLab/env_isaaclab/bin/python \
  scripts/run_m2_two_g1_box_push.py \
  --num_envs 1 \
  --max_steps 130 \
  --policy-path results/m2_proxy_training/halypo_policy.json \
  --stop-on-success \
  --viz newton
```
