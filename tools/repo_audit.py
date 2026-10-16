#!/usr/bin/env python3
"""Inventory code, tests and result metadata without importing experiments.

This is deliberately not a garbage collector: an unreferenced failure may be
an ancestor of a selected run. Inventory candidates require human review.
Neither media/checkpoints nor credentials are opened; symlinks are not followed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_inventory(root: Path) -> dict:
    """Report byte duplicates and shadowed definitions, not 'old = redundant'."""
    by_hash = defaultdict(list)
    shadowed = []
    counts = {}
    for path in sorted((root / 'tests').rglob('test_*.py')):
        raw = path.read_bytes()
        relative = str(path.relative_to(root))
        by_hash[hashlib.sha256(raw).hexdigest()].append(relative)
        tree = ast.parse(raw)
        names = defaultdict(list)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
                names[node.name].append(node.lineno)
        counts[relative] = sum(len(lines) for lines in names.values())
        shadowed.extend({'file': relative, 'name': name, 'lines': lines}
                        for name, lines in names.items() if len(lines) > 1)
    return {'files': len(counts), 'top_level_test_definitions': sum(counts.values()),
            'byte_identical_files': [v for v in by_hash.values() if len(v) > 1],
            'shadowed_definitions': shadowed, 'definitions_per_file': counts}


def result_inventory(root: Path) -> dict:
    """Stat every deployment artifact, grouping by evidence tree/campaign.

    Logical byte totals count hardlinks separately. This is not disk-block
    usage, a validation of contents, or proof that an experiment is complete.
    """
    base = root / 'results' / 'deployment'
    groups = defaultdict(lambda: {'files': 0, 'bytes': 0, 'symlinks': 0, 'extensions': Counter()})
    empty_logs, errors = [], []
    for directory, dirs, files in os.walk(base, followlinks=False, onerror=lambda e: errors.append(str(e))):
        here = Path(directory)
        for name in [*dirs, *files]:
            path = here / name
            parts = path.relative_to(base).parts
            group = '/'.join(parts[:2]) if parts[0] in {'formal', 'non_paper', 'paper_candidates'} else parts[0]
            row = groups[group]
            try:
                if path.is_symlink():
                    row['symlinks'] += 1
                    continue
                if name in dirs:
                    continue
                size = path.stat().st_size
            except OSError as exc:
                errors.append(f'{path}: {exc}')
                continue
            row['files'] += 1
            row['bytes'] += size
            row['extensions'][path.suffix or '(none)'] += 1
            if size == 0 and path.suffix == '.log':
                empty_logs.append(str(path.relative_to(root)))
    return {'root': str(base), 'groups': dict(sorted(groups.items())),
            'files': sum(g['files'] for g in groups.values()),
            'bytes': sum(g['bytes'] for g in groups.values()),
            'empty_log_candidates': sorted(empty_logs), 'errors': errors,
            'deletion_authorized': False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write a new report; never overwrite an earlier audit')
    args = parser.parse_args()
    result = {'schema_version': 1, 'tests': test_inventory(ROOT), 'results': result_inventory(ROOT)}
    data = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(data)
        print(json.dumps({'report': str(args.output), 'result_files': result['results']['files'],
                          'result_bytes': result['results']['bytes'], 'test_files': result['tests']['files']}))
    else:
        print(data, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
