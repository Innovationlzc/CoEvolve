# Environment setup

The offline SDK and live motor stack have separate dependencies. Install the
offline environment first; a successful smoke test does not qualify physics.

## Python environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m coevolve_experiments adapter-smoke --method all
python -m pytest -q tests/maintenance tests/experiments/test_cli.py
```

Python 3.10+ is required. Linux is required for the current live deployment.
On Debian/Ubuntu, install `python3-venv` if environment creation reports a
missing `ensurepip` module.
Extras: `sim` for Python simulation clients and `local-vlm` for local model
support. Installing `sim` does not install
the native SONIC stack. For local models, install a hardware-compatible
PyTorch/CUDA build separately and obtain the configured weights; remote model
clients do not need a local model download.

The wheel includes the static Skill SDK catalog, not scenes or evidence.
Scene validation outside an editable checkout additionally needs
`--repo-root ../CoEvolve` (relative to your working directory).

## SONIC and MuJoCo

Acquire [GR00T WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
and its authorized policy/planner weights using the upstream installation
instructions. The expected local location is
`third_party/GR00T-WholeBodyControl`, with its simulation interpreter at
`.venv_sim/bin/python`. That external tree and all weights are ignored by Git.
Do not move an existing Isaac Lab environment into this repository.

The two-G1 stack needs both DDS topic-prefix support and incremental heading
commands. From the CoEvolve root, check the patch state before applying:

```bash
export COEVOLVE_ROOT="$PWD"
cd "$COEVOLVE_ROOT/third_party/GR00T-WholeBodyControl"
git apply --check "$COEVOLVE_ROOT/patches/gear_sonic_multi_robot_dds_prefix.patch"
git apply --check "$COEVOLVE_ROOT/patches/gear_sonic_heading_increment.patch"
```

If these checks succeed and the changes are not already installed, apply those
two patches, then rebuild using the upstream environment:

```bash
git apply "$COEVOLVE_ROOT/patches/gear_sonic_multi_robot_dds_prefix.patch"
git apply "$COEVOLVE_ROOT/patches/gear_sonic_heading_increment.patch"
cd gear_sonic_deploy
source scripts/setup_env.sh
just --justfile .justfile build
```

A failed patch check needs inspection, not forced application: the patch may
already be present or the upstream revision may differ. Both the native deploy
binary and MuJoCo bridge must support DDS prefixes; script flags alone do not
provide isolation. The selected upstream revision, weights and patch hashes
must be recorded for a physical run.

For the CoEvolve planner environment:

```bash
cd "$COEVOLVE_ROOT"
source .venv/bin/activate
python -m pip install -e '.[dev,sim]'
```

Keep the upstream simulation environment separate. Live startup also needs
working rendering, native DDS and the upstream motor dependencies; it has not
been validated by the portable Python checks alone.

## Foundation model provider

The `openai_compatible` backend accepts a user-configured chat-completions
service. No provider endpoint or credential is bundled. Configure:

```text
--planner-backend openai_compatible
--planner-model <exact-provider-model-id>
--planner-base-url <your-HTTPS-base-URL>
--planner-api-key-env COEVOLVE_API_KEY
```

Supply `COEVOLVE_API_KEY` through the shell or a secret manager; never put the
secret value in commands, Git, prompts or manifests. Python entrypoints do not
automatically load `.env.local`. Requests may send scene images and incur fees.
`COEVOLVE_BASE_URL` supplies the endpoint if the CLI option is omitted. An absent
or invalid endpoint fails before sending a request. Billing lookup is disabled
unless an explicit same-origin `usage_path_template` is configured on the client.

For local inference, use `--planner-backend local_qwen_vl` and the model
configuration under `src/praxis_coop/configs/model_registry.py`. Review model
code/revisions before loading: the current local loader can trust remote code.
Public protocol routes use reserved example domains for offline validation.
They are not the original sealed route identities. Historical continuations
and their frozen runtimes are not distributed; edits to public source do not
requalify them. Never rewrite evidence hashes to make a check pass.

Filesystem locations are relative to the checkout or configured at runtime.
The optional Isaac Lab Kit app expects an adjacent `../IsaacLab/source` tree;
adjust its extension folder setting if your external runtime is elsewhere.

Continue with [the run guide](docs/REPRODUCTION.md).
