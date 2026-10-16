"""Offline regressions for publication and provider configuration."""
import pytest

from tools.check_public_tree import inspect
from praxis_coop.llm import openai_compatible_client as client_module
from praxis_coop.llm.errors import ProviderFatalError


def test_private_path_and_service_findings_do_not_include_matched_values():
    data = ("# /" + "home" + "/private-user/project https://" + "private-gateway.internal/v1").encode()
    assert inspect("config.py", data) == {"personal-absolute-path", "unreviewed-service-host"}
    assert inspect("config.py", b"# https://example.invalid/v1") == set()
    assert inspect("results/run.json", b"{}") == {"excluded-tree"}


@pytest.mark.parametrize("endpoint", ["", "http://remote.example/v1", "https://" + "user:secret@example.invalid/v1",
                                      "https://example.invalid/v1?key=secret"])
def test_invalid_endpoint_fails_before_network(monkeypatch, endpoint):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "fixture-only")
    monkeypatch.setattr(client_module.request, "urlopen", lambda *_a, **_k: pytest.fail("network attempted"))
    client = client_module.OpenAICompatibleTextClient(model_name="fixture", base_url=endpoint,
                                                    api_key_env="TEST_PROVIDER_KEY")
    with pytest.raises(ProviderFatalError):
        client.generate(system="fixture", user="fixture")


def test_environment_endpoint_is_resolved_when_client_is_created(monkeypatch):
    monkeypatch.setenv("COEVOLVE_BASE_URL", "https://example.invalid/v1")
    assert client_module.OpenAICompatibleTextClient(model_name="fixture").base_url == "https://example.invalid/v1"


def test_billing_is_opt_in_and_rejects_cross_origin_template(monkeypatch):
    monkeypatch.setattr(client_module.request, "urlopen", lambda *_a, **_k: pytest.fail("network attempted"))
    args = dict(base_url="https://example.invalid/v1", api_key="fixture", request_id="fixture", timeout_s=1)
    assert client_module._fetch_openai_compatible_request_usage(**args) == {"_lookup_status": "not_attempted"}
    for template in ("https://elsewhere.example/{request_id}", "//elsewhere.example/{request_id}", "/usage/{wrong}"):
        with pytest.raises(ProviderFatalError):
            client_module._fetch_openai_compatible_request_usage(**args, usage_path_template=template)


def test_embedded_operator_quotes_are_not_runtime_prompts():
    text = b"AUDIT = {'user_instruction': 'synthetic private operator note'}"
    assert inspect('script.py', text) == {'embedded-operator-request'}
    assert inspect('script.py', b"TASK = {'instruction': 'Carry the box'}") == set()


def test_history_scan_detects_removed_content_and_retired_commit(tmp_path):
    import subprocess
    from tools.check_public_tree import inspect_history

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path).decode().strip()

    git('init', '--quiet')
    git('config', 'user.name', 'Synthetic Fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    path = tmp_path / 'config.txt'
    path.write_text('https://' + 'retired-service.internal/v1\n')
    git('add', 'config.txt')
    git('commit', '-qm', 'synthetic old configuration')
    retired = git('rev-parse', 'HEAD')
    path.write_text('https://example.invalid/v1\n')
    git('add', 'config.txt')
    git('commit', '-qm', 'synthetic cleaned configuration')
    report = inspect_history(tmp_path, 'HEAD', {'commits': [retired]})
    assert report['commits'] == 2
    assert report['file_versions'] == 2
    rules = {rule for row in report['findings'] for rule in row['rules']}
    assert {'retired-history-commit', 'unreviewed-service-host'} <= rules


def test_historical_audit_has_no_shipped_default_pins(tmp_path):
    from coevolve_experiments.production import audit_frozen_v83_checkout, FrozenV83AuditError
    from coevolve_experiments.reporting import audit_table1_source_cohort, Table1SourceAuditError

    with pytest.raises(TypeError):
        audit_frozen_v83_checkout(tmp_path)
    with pytest.raises(FrozenV83AuditError, match='hashes are required'):
        audit_frozen_v83_checkout(tmp_path, expected_commit='a' * 40, expected_file_hashes={})
    with pytest.raises(Table1SourceAuditError, match='not distributed'):
        audit_table1_source_cohort(tmp_path / 'absent.json', None)
