#!/usr/bin/env python3
"""Run public software tests in fresh interpreters.

Core runner tests can mutate module-global state. One subprocess per file
prevents cross-file contamination. In a Git checkout, ignored local audits are
not selected automatically. Explicit paths can still run a local regression.
No failure is converted to a skip; external-runtime tests require --integration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def selected_files(suite: str) -> list[Path]:
    directory = ROOT / 'tests'
    if suite in {'contracts', 'maintenance'}:
        directory /= 'experiments' if suite == 'contracts' else 'maintenance'
    candidates = sorted(directory.rglob('test_*.py'))
    if (ROOT / '.git').exists():
        # Include reviewed new tests, but not ignored historical files still on
        # the maintainer's disk. A source archive has only published files.
        selected = subprocess.check_output(
            ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', 'tests/'],
            cwd=ROOT,
        ).decode().split('\0')
        paths = {ROOT / path for path in selected if path}
        candidates = [path for path in candidates if path in paths]
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('all', 'contracts', 'maintenance'), default='all')
    parser.add_argument('--list', action='store_true', help='inventory without importing test modules')
    parser.add_argument('--integration', action='store_true', help='also run tests requiring external SONIC assets')
    parser.add_argument('--timeout', type=float, default=300, help='per-file timeout in seconds')
    parser.add_argument('files', nargs='*', type=Path, help='optional explicit test files within this repo')
    args = parser.parse_args()
    files = args.files or selected_files(args.suite)
    resolved = [p.resolve() for p in files]
    if args.timeout <= 0 or not all(p.is_file() and p.is_relative_to(ROOT / 'tests') for p in resolved):
        parser.error('positive timeout and existing files under tests/ required')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    # Start clean: inherited paths can point at a different frozen source tree.
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT)])
    failed = []
    for path in resolved:
        relative = path.relative_to(ROOT)
        print(f'\n{relative}', flush=True)
        if args.list:
            continue
        try:
            command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider']
            if not args.integration:
                command.extend(['-m', 'not integration'])
            child = subprocess.Popen(command + [str(path)], cwd=ROOT, env=environment, start_new_session=True)
            if child.wait(timeout=args.timeout):
                failed.append(str(relative))
        except subprocess.TimeoutExpired:
            # Terminate only this test process group, never unrelated workers.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            failed.append(f'{relative} (timeout)')
        except KeyboardInterrupt:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
    if not args.list:
        print(f'\nFiles: {len(resolved)}; passed: {len(resolved) - len(failed)}; failed: {len(failed)}')
        for item in failed:
            print(f'FAILED {item}')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
