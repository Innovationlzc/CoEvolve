# Results Layout

Current result outputs are split into:

- `deployment/`: current physical/deployment experiment outputs only.
- `spec_memory/static/`: curated reusable cooperation specifications.
- `spec_memory/dynamic/`: Qwen-generated specifications learned from verifier-gated rollouts.

Old proxy, failed, and sweep outputs were removed from the active result tree so
they are not accidentally used as paper evidence.
