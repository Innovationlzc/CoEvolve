"""Exercise publication rules without changing the real Git index or evidence."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_GUIDES = (
    "README.md", "ENV_SETUP.md", "CONTRIBUTING.md", "tests/README.md",
    "docs/API.md", "docs/ARCHITECTURE.md", "docs/REPRODUCTION.md", "docs/BASELINES.md",
)


def ignored_paths(tmp_path: Path, paths: list[str]) -> set[str]:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=/dev/null", "check-ignore", "--no-index", "--stdin"],
        cwd=tmp_path,
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    return set(result.stdout.splitlines())


def test_code_required_docs_and_exact_figures_are_publishable(tmp_path):
    allowed = [
        "src/praxis_coop/cap_policy/policy_executor.py",
        "scripts/run_cap_box_carry_evolution.py",
        "tests/maintenance/test_publish_scope.py",
        "configs/example.json",
        ".github/workflows/offline-checks.yml",
        ".env.example",
        *PUBLIC_GUIDES,
        "assets/manual_scenes/two_g1_box_push_layout.usda",
        "assets/generated_scenes/narrow.usda.receipt.json",
        "src/praxis_coop/data/default_specs.jsonl",
        "docs/assets/figures/manifest.json",
        *[f"docs/assets/figures/{name}.png" for name in (
            "teaser", "framework", "baseline_comparison", "cross_layouts",
            "repair_case", "layout_overview",
        )],
    ]
    assert ignored_paths(tmp_path, allowed) == set()


def test_paper_records_secrets_and_unreviewed_media_stay_local(tmp_path):
    excluded = [
        "source/ICLR27_CoEvolve.zip",
        "source/README.md",
        "source/g1_box_carry_cap_task_brief.md",
        "AGENTS.md", "TODO.md", "scripts/README.md",
        "src/coevolve_experiments/README.md",
        "src/coevolve_experiments/production/README.md",
        "docs/README.md", "docs/PUBLISHING.md", "docs/FIGURES.md",
        "docs/OPEN_SOURCE_READINESS.md", "docs/PARALLEL_EXECUTION.md",
        "docs/SONIC_INTEGRATION.md", "docs/STATIC_SONIC_SKILLS.md",
        "docs/EVOLUTION_EXPERIMENT.md",
        "docs/CAP_AGENT0_ADAPTATION_20260914.md",
        "docs/ASPIRE_ADAPTATION_20260914.md",
        "source/CoEvolve/main.tex",
        "source/praxis_coop_method_aaai.tex",
        "source/CoEvolve/results_20260914/summary.json",
        "source/CoEvolve/metrics_update_20260915/table1.tex",
        "source/CoEvolve/timing_update_20260924/runs.csv",
        "docs/TABLE34_CLOSEOUT_STATUS.md",
        "docs/REPOSITORY_MAINTENANCE.md",
        "docs/new_internal_note.md",
        "docs/assets/figures/unreviewed.png",
        "docs/assets/figures/old.pdf",
        "docs/assets/figures/drafts/test.png",
        "docs/assets/other.png",
        "screenshot.png", "paper.pdf", "slides.pptx", "backup.zip",
        ".env", ".env.local", ".env.production",
        "private.pem", "private.key", "identity.p12", "id_ed25519",
        ".vscode/settings.json",
        ".agents/local.json", ".codex/local.toml",
        "results/README.md",
        "results/deployment/README.md",
        "results/deployment/static_skill_success_manifest.json",
        "results/spec_memory/static/default_specs.jsonl",
        "results/spec_memory/dynamic/README.md",
        "results/new_summary.csv",
        "results/deployment/run/metrics.json",
        "results/deployment/run/frame.png",
        "models/model.safetensors", "third_party/runtime/main.py",
        "HaLyPO/old_run.json", ".cleanup-quarantine/old.md",
        ".venv/pyvenv.cfg",
        "scripts/new_one_off.py", "scripts/nested/internal.py",
        "scripts/prepare_final_table34_v2.py",
        "scripts/layout4_corridor_runtime_v1.py",
        "tests/test_corrected_table34_v3.py",
        "tests/fixtures/private_record.json",
        "tests/experiments/test_manuscript_audit.py",
    ]
    assert ignored_paths(tmp_path, excluded) == set(excluded)


def test_script_catalog_matches_exact_public_allowlist(tmp_path):
    rules = (ROOT / '.gitignore').read_text().splitlines()
    allowed = {line[2:] for line in rules if line.startswith('!/scripts/')}
    assert len(allowed) == 18
    assert all((ROOT / path).is_file() for path in allowed)
    assert ignored_paths(tmp_path, sorted(allowed)) == set()
    guide = (ROOT / 'docs/REPRODUCTION.md').read_text().split('## Script catalog', 1)[1]
    names = set(re.findall(r'`([a-z0-9_]+\.(?:py|sh))`', guide))
    assert names == {Path(path).name for path in allowed}


def test_public_guides_link_only_to_included_files(tmp_path):
    targets = set()
    for guide in PUBLIC_GUIDES:
        document = ROOT / guide
        for link in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", document.read_text()):
            if "://" in link or link.startswith(("#", "mailto:")):
                continue
            target = (document.parent / link.split("#", 1)[0]).resolve()
            assert target.is_file(), (guide, link)
            targets.add(target.relative_to(ROOT).as_posix())
    assert ignored_paths(tmp_path, sorted(targets)) == set()


def test_approved_figure_previews_match_their_provenance_manifest():
    """Verify shipped previews without requiring the excluded manuscript ZIP."""
    root = ROOT / "docs/assets/figures"
    manifest = json.loads((root / "manifest.json").read_text())
    assert {row["image"] for row in manifest["figures"]} == {
        "teaser.png", "framework.png", "baseline_comparison.png",
        "cross_layouts.png", "repair_case.png", "layout_overview.png",
    }
    for row in manifest["figures"]:
        data = (root / row["image"]).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        assert hashlib.sha256(data).hexdigest() == row["png_sha256"]
        assert len(data) == row["png_bytes"]
        assert struct.unpack(">II", data[16:24]) == (row["width_px"], row["height_px"])
        assert row["width_px"] == 1800
