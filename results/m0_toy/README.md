# M0 Toy Result

Command:

```bash
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m0_toy.py --steps 300 --output-dir results/m0_toy
```

Summary:

| Method | Initial V | Final V | Update alignment | Conflict rate | Constraint violation |
|---|---:|---:|---:|---:|---:|
| independent | 5.8457 | 12.3329 | ~0.0000 | 0.78 | 1.00 |
| HALyPO | 5.8457 | 0.0307 | 0.1961 | 0.00 | 0.00 |

The toy game confirms the M0 target: the independent rotational field violates
the Lyapunov decrease condition, while HALyPO projects the update direction so
`V(theta)` contracts with zero measured constraint violations.

