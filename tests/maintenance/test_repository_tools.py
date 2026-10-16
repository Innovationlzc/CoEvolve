"""Test maintenance against temporary trees, never against live results."""
import importlib.util
from pathlib import Path

import pytest


def load_tool(name):
    path = Path(__file__).resolve().parents[2] / 'tools' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_inventory_does_not_follow_external_symlinks(tmp_path):
    audit = load_tool('repo_audit')
    base = tmp_path / 'results/deployment/formal/run'
    base.mkdir(parents=True)
    (base / 'metrics.json').write_text('{}')
    (base / 'outside').symlink_to('/etc', target_is_directory=True)
    (base / 'empty.log').touch()
    report = audit.result_inventory(tmp_path)
    assert report['files'] == 2
    assert report['bytes'] == 2
    assert report['groups']['formal/run']['symlinks'] == 1
    assert report['deletion_authorized'] is False
    assert report['errors'] == []


def test_quarantine_allowlist_preserves_results_and_lock_files(tmp_path):
    cleanup = load_tool('quarantine_artifacts')
    for name in ('build', 'src/pkg/__pycache__', 'results/deployment/formal/run'):
        (tmp_path / name).mkdir(parents=True)
    result = tmp_path / 'results/deployment/formal/run'
    (result / 'stderr.log').touch()
    (result / 'queue.lock').touch()
    (result / 'metrics.json').write_text('{}')
    targets = {str(p.relative_to(tmp_path)) for p in cleanup.candidates(tmp_path)}
    assert targets == {'build', 'src/pkg/__pycache__'}


def test_cleanup_refuses_symlinks_in_generated_output(tmp_path):
    cleanup = load_tool('quarantine_artifacts')
    path = tmp_path / 'build'
    path.mkdir()
    (path / 'reference').symlink_to('/etc/hosts')
    with pytest.raises(ValueError, match='symlink'):
        cleanup.inventory(tmp_path, [path])


def test_nonempty_operator_log_is_never_a_cleanup_candidate(tmp_path):
    cleanup = load_tool('quarantine_artifacts')
    path = tmp_path / cleanup.EMPTY_LOGS[0]
    path.parent.mkdir(parents=True)
    path.write_text('an actual failure record')
    assert path not in cleanup.candidates(tmp_path)


def test_duplicate_test_inventory_reports_only_exact_files(tmp_path):
    audit = load_tool('repo_audit')
    tests = tmp_path / 'tests'
    tests.mkdir()
    for name in ('test_a.py', 'test_b.py'):
        (tests / name).write_text('def test_same():\n    assert True\n')
    report = audit.test_inventory(tmp_path)
    assert report['byte_identical_files'] == [['tests/test_a.py', 'tests/test_b.py']]
    assert report['shadowed_definitions'] == []
