"""Portable tests: no frozen checkout, model credentials, or simulator."""
from pathlib import Path
import sys
import os
import subprocess
import threading
import time

import pytest

from coevolve_experiments.cli import build_parser
from coevolve_experiments.methods.cap_agent0 import CaPAgent0Adapter


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PYTHON_SCRIPTS = [
    line[2:] for line in (ROOT / '.gitignore').read_text().splitlines()
    if line.startswith('!/scripts/') and line.endswith('.py')
]


@pytest.mark.parametrize('script', PUBLIC_PYTHON_SCRIPTS)
def test_public_entrypoint_help_needs_no_provider_or_simulator(script):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run(
        [sys.executable, str(ROOT / script), '--help'], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, script + '\n' + result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('workers', [1, 3, 9])
def test_candidate_parallelism_is_bounded_without_changing_candidate_count(workers):
    lock = threading.Lock()
    active, peak = 0, 0
    calls = []
    candidate_group = threading.Barrier(workers)

    def invoke(role, payload):
        nonlocal active, peak
        with lock:
            calls.append((role, payload['request_id']))
            active += 1
            peak = max(peak, active)
        if role == 'cap_candidate':
            candidate_group.wait(timeout=2)
        time.sleep(.005)
        with lock:
            active -= 1
        return {'content': 'pass' if role != 'cap_visual_difference' else 'initial scene'}

    adapter = CaPAgent0Adapter(invoke, candidate_workers=workers)
    result = adapter.propose({}, images=['mock.png'])
    assert len(calls) == 11  # one VDM + nine candidates + one synthesis
    assert [row['index'] for row in result['candidates']] == list(range(1, 10))
    assert peak == workers
    assert result['candidate_count'] == 9
    assert all(row['physically_executed'] is False for row in result['candidates'])


@pytest.mark.parametrize('workers', [0, 10, -1, 1.5, True])
def test_invalid_worker_limit_rejected_before_calls(workers):
    with pytest.raises(ValueError, match='candidate_workers'):
        CaPAgent0Adapter(lambda *_: pytest.fail('no request allowed'), candidate_workers=workers)


def test_cli_compatibility_facade_binds_all_registered_commands():
    parser = build_parser()
    assert callable(parser.parse_args(['adapter-smoke']).handler)
    assert callable(parser.parse_args(['validate-scenes']).handler)
    assert parser.parse_args(['validate-scenes', '--repo-root', '/tmp/repo']).repo_root == Path('/tmp/repo')


def test_scene_cli_passes_explicit_asset_root_without_relocating_evidence(monkeypatch, tmp_path, capsys):
    from coevolve_experiments import cli
    from types import SimpleNamespace

    roots = []

    def validate(scene, **kwargs):
        roots.append(kwargs['repo_root'])
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(cli, 'validate_scene', validate)
    serialize = cli.to_primitive
    monkeypatch.setattr(cli, 'to_primitive', lambda value: (
        {'ok': value.ok} if isinstance(value, SimpleNamespace) else serialize(value)
    ))
    assert cli.main(['validate-scenes', '--repo-root', str(tmp_path)]) == 0
    assert roots and all(root == tmp_path for root in roots)


def test_packaged_baseline_imports_do_not_change_import_paths():
    import importlib

    before = list(sys.path)
    module = importlib.import_module('coevolve_experiments.methods.cap_agent0')
    importlib.import_module('coevolve_experiments.methods.aspire')
    importlib.import_module('coevolve_experiments.methods.team_program')
    assert sys.path == before
    assert module.TEMPERATURES == tuple(i / 10 for i in range(1, 10))
    assert module.Adapter is module.CaPAgent0Adapter


def test_source_and_wheel_use_packaged_catalog_without_results(tmp_path, monkeypatch):
    from praxis_coop.cap_policy import static_specs

    root = Path(__file__).resolve().parents[2]
    packaged = root / 'src/praxis_coop/data/default_specs.jsonl'
    assert static_specs.STATIC_SKILL_CATALOG_PATH == packaged
    assert {record['family'] for record in static_specs.load_static_skill_catalog()} == (
        static_specs.REQUIRED_SKILL_FAMILIES
    )
    # Both an editable checkout and an installed wheel resolve module-local data.
    # A missing resource must not fall back to a historical results directory.
    module = tmp_path / 'praxis_coop/cap_policy/static_specs.py'
    monkeypatch.setattr(static_specs, '__file__', str(module))
    assert static_specs._catalog_path() == tmp_path / 'praxis_coop/data/default_specs.jsonl'


def test_missing_packaged_catalog_still_fails_closed(tmp_path, monkeypatch):
    from praxis_coop.cap_policy import static_specs

    static_specs.load_static_skill_catalog.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(static_specs, 'STATIC_SKILL_CATALOG_PATH', tmp_path / 'missing.jsonl')
            with pytest.raises(RuntimeError, match='missing static skill catalog'):
                static_specs.load_static_skill_catalog()
    finally:
        static_specs.load_static_skill_catalog.cache_clear()


def test_formal_manifest_hashes_shipped_catalog_without_result_inputs(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace

    from praxis_coop.cap_policy.contracts import BoxCarryTask
    from praxis_coop.cap_policy.formal_eval import (
        FORMAL_BACKBONES, FORMAL_EVAL_PROTOCOL_ID, write_formal_run_manifest,
    )
    from praxis_coop.cap_policy.static_specs import STATIC_SKILL_CATALOG_PATH

    assert not (tmp_path / 'results').exists()
    args = SimpleNamespace(
        eval_protocol=FORMAL_EVAL_PROTOCOL_ID,
        output_dir=tmp_path / 'run',
        planner_model=FORMAL_BACKBONES[0].requested_model,
        planner_backend=FORMAL_BACKBONES[0].planner_backend,
        run_index=0, seed=0, physical_safety_profile='task',
    )
    manifest = write_formal_run_manifest(
        args=args, repo_root=tmp_path, codebase_snapshot={},
        task=BoxCarryTask(task_id='catalog-test', target='target'),
        status='initialized',
    )
    catalog_hash = hashlib.sha256(STATIC_SKILL_CATALOG_PATH.read_bytes()).hexdigest()
    expected = hashlib.sha256(json.dumps(
        {'default_specs.jsonl': catalog_hash}, sort_keys=True, separators=(',', ':'),
    ).encode()).hexdigest()
    assert json.loads(manifest.read_text())['artifact_hashes']['static_memory_sha256'] == expected


def test_deploy_launcher_resolves_its_checkout_without_starting_motor_stack(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[2]
    script = root / 'scripts/run_official_sonic_deploy_zmq_manager.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    # Execute only the root-resolution preamble in an unrelated checkout.
    # No external runtime, network, build or motor process is touched.
    checkout = tmp_path / 'another checkout'
    probe = checkout / 'scripts/root_probe.sh'
    probe.parent.mkdir(parents=True)
    preamble = script.read_text().split('DEPLOY_DIR=', 1)[0]
    probe.write_text(preamble + 'printf "%s\\n" "$REPO_ROOT"\n')
    result = subprocess.run(['bash', str(probe)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(checkout)
