# Test profiles

Install `.[dev]`. These commands do not launch physical experiments.

## Portable checks

```bash
python -m pytest -q tests/maintenance tests/experiments/test_cli.py
python -m coevolve_experiments validate-scenes --mode planning
python -m coevolve_experiments adapter-smoke --method all
python tools/check_public_tree.py
```

This is the CI profile: no credentials, simulator, `results/` or historical
worktrees are needed. It checks interfaces, publication rules and offline
contracts, not model quality or robot performance.

## Public regressions

Video-recording tests also require OpenCV. If `cv2` is not already provided by
your simulation environment, install `opencv-python-headless` first.

```bash
python tools/run_tests.py --list
python tools/run_tests.py --suite all
python tools/run_tests.py tests/test_model_role_pipeline.py
```

Each file runs in a fresh interpreter to isolate module-global state. In a Git
checkout the runner selects only published or unignored new tests, not local
historical audits. It returns nonzero on failures or timeouts.

The published tests use synthetic fixtures for provider responses, accounting
and protocol contracts. Baseline mechanism tests import the installed package;
they do not require historical launch wrappers or frozen sibling checkouts.
Original manuscript/campaign audits and recorded model-output fixtures stay
local with their evidence; they are not part of this distribution.
Historical cohort pins are also excluded. Public tests verify that their audit
entrypoints fail closed rather than substitute fabricated provenance.

## Optional integration checks

`python tools/run_tests.py --integration` also enables the `integration` marker,
including scene generation from the separately installed upstream SONIC XML.
Without this flag, integration tests are explicitly deselected, not treated as
passing. The external runtime and licensed assets are not included. Neither
profile runs a live robot experiment or contacts a model provider.
