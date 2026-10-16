# Contributing

Start with [setup](ENV_SETUP.md), [architecture](docs/ARCHITECTURE.md) and
[tests](tests/README.md). Project licensing and contribution terms are pending;
do not infer a license from repository access.

## Checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/maintenance tests/experiments/test_cli.py
python -m coevolve_experiments validate-scenes --mode planning
python -m coevolve_experiments adapter-smoke --method all
python tools/check_public_tree.py --staged
python tools/check_public_tree.py --ref HEAD
git diff --check
```

CI is offline. Provider calls and physics require explicit operator opt-in.
Run the full public regression suite with `python tools/run_tests.py`.
Add `--integration` only when the separate SONIC assets are installed.

## Change boundaries

- Put reusable logic in `src/` and entrypoints in `scripts/`; add focused tests.
- New script entrypoints need an explicit `.gitignore` allowlist entry and a
  short catalog entry in the run guide. Keep one-off local tools unpublished.
- Keep behavior-preserving refactors separate from experiment protocol changes.
- Preserve source identities, cumulative budgets, checkpoint ancestry and raw outcomes.
- Never commit credentials, `results/`, runtimes, weights or manuscript bundles.
- Test model-request parallelism separately from concurrent physical stacks.
- Include purpose, behavior changes, tests run and limitations in each PR.

Do not post credentials or sensitive exploit details in public issues.
A private security contact and redistribution terms must be established before
a formal open-source release.

The publication guard checks only Git-selected files and prints rule IDs, not
matched secrets. Review diffs and binary metadata as well; automated scanning
cannot certify that every possible secret has been removed. Do not merge or
push pre-cleanup branches into the clean public history.
The history check covers reachable committed versions, not GitHub-managed PR
refs, detached cached objects, Actions artifacts or other people's clones.
