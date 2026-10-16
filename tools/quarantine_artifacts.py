#!/usr/bin/env python3
"""Recoverably remove reviewed build/test caches and two empty operator logs.

Default is a dry run. No run directory, result JSON, checkpoint, media, memory,
lock, installed egg-info, external runtime or credential file is eligible.
This narrow allowlist intentionally does not treat non_paper/ as disposable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EMPTY_LOGS = (
    'results/deployment/non_paper/table4-parallel-isolation/campaign.log',
    'results/deployment/non_paper/table4-parallel-isolation/boundary_drain.log',
)


def candidates(root: Path) -> list[Path]:
    result = [root / name for name in ('build', '.pytest_cache', '.ruff_cache')]
    for name in ('src', 'scripts', 'tests', 'tools'):
        for directory, dirs, _ in os.walk(root / name, followlinks=False):
            if '__pycache__' in dirs:
                result.append(Path(directory) / '__pycache__')
                dirs.remove('__pycache__')
    result.extend(root / name for name in EMPTY_LOGS
                  if (root / name).is_file() and (root / name).stat().st_size == 0)
    return sorted(path for path in result if path.exists())


def inventory(root: Path, paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        leaves = [path, *path.rglob('*')] if path.is_dir() else [path]
        for leaf in leaves:
            if leaf.is_symlink() or not leaf.resolve().is_relative_to(root.resolve()):
                raise ValueError(f'refusing symlink or escaped path: {leaf}')
            if leaf.is_file():
                rows.append({'path': str(leaf.relative_to(root)), 'bytes': leaf.stat().st_size,
                             'sha256': hashlib.sha256(leaf.read_bytes()).hexdigest()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--restore', type=Path, help='explicit manifest created by an earlier apply')
    parser.add_argument('--batch', default='20261016', help='new quarantine subdirectory name')
    args = parser.parse_args()
    if args.restore:
        manifest = args.restore.resolve()
        if not manifest.is_relative_to((ROOT / '.cleanup-quarantine').resolve()):
            parser.error('restore manifest must be inside the local quarantine')
        plan = json.loads(manifest.read_text())
        if plan['root'] != str(ROOT):
            parser.error('manifest belongs to another repository')
        # Verify all destinations before moving anything; never overwrite new work.
        for name in plan['targets']:
            if Path(name).is_absolute() or '..' in Path(name).parts or not Path(name).parts:
                parser.error('invalid restore path')
            if (ROOT / name).exists() or (ROOT / name).is_symlink():
                parser.error(f'restore destination already exists: {name}')
            source = manifest.parent / name
            if not source.exists() or source.is_symlink() or not source.resolve().is_relative_to(manifest.parent):
                parser.error(f'invalid quarantined source: {name}')
        for row in plan['files']:
            source = manifest.parent / row['path']
            if source.is_symlink() or not source.resolve().is_relative_to(manifest.parent):
                parser.error('invalid file in restore manifest')
            if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
                parser.error(f'quarantine content changed: {row["path"]}')
        for name in plan['targets']:
            (ROOT / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(manifest.parent / name), str(ROOT / name))
        print('Restored all targets without overwriting existing files.')
        return 0
    if Path(args.batch).name != args.batch or args.batch in {'', '.', '..'}:
        parser.error('batch must be one directory name')
    paths = candidates(ROOT)
    files = inventory(ROOT, paths)
    plan = {'root': str(ROOT), 'targets': [str(p.relative_to(ROOT)) for p in paths],
            'files': files, 'bytes': sum(f['bytes'] for f in files)}
    print(json.dumps({k: v for k, v in plan.items() if k != 'files'}, indent=2))
    if args.apply:
        destination = ROOT / '.cleanup-quarantine' / args.batch
        destination.mkdir(parents=True, exist_ok=False)
        # Write the recovery manifest before moving even the first file.
        (destination / 'manifest.json').write_text(json.dumps(plan, indent=2) + '\n')
        for path in paths:
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        print(f'Quarantined {len(files)} files; recovery manifest: {destination / "manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
