# M1 Transport Result

Command:

```bash
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m1_transport.py --output-dir results/m1_transport
```

Summary:

| Method | Initial return | Final return | Initial V | Final V | Conflict rate | Constraint violation |
|---|---:|---:|---:|---:|---:|---:|
| independent | -1.9592 | -0.2615 | 0.6416 | 3.8922 | 0.15 | 1.00 |
| HALyPO | -1.9585 | -0.3258 | 0.6416 | 0.3653 | 0.10 | 0.00 |

M1 is a 2D policy-gradient proxy, not full HAPPO yet. It confirms that HALyPO
can improve the team objective while keeping rationality-gap dynamics stable.
Independent updates reach the target faster in this small toy, but their gap and
Lyapunov constraint violations grow sharply.

Use `--show` to open the saved plot after the run on a desktop session:

```bash
/home/admin1/Project/IsaacLab/env_isaaclab/bin/python scripts/run_m1_transport.py --show
```

