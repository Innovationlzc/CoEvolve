#!/usr/bin/env python3
"""Check only Git-selected files; never read ignored evidence or credentials.

Findings contain filenames and rule IDs, not matched secret material. This is
a publication guard, not a guarantee that arbitrary prose contains no secrets.
Use --staged before publication to inspect the exact bytes in the Git index.
Use --ref to inspect reachable history. An optional private denylist can bind
known incident identifiers without publishing them in scanner source code.
"""
from __future__ import annotations

import argparse
import ast
import ipaddress
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_HOSTS = {
    "github.com", "arxiv.org", "nvlabs.github.io", "www.w3.org", "purl.org",
    "schemas.openxmlformats.org", "openaipublic.blob.core.windows.net",
    "omniverse-content-production.s3.us-west-2.amazonaws.com",
    "omniverse-content-staging.s3-us-west-2.amazonaws.com",
}
EXCLUDED_ROOTS = {"results", "source", "third_party", "models", "HaLyPO",
                  ".local", ".cleanup-quarantine", ".agents", ".codex"}
PERSONAL_PATH = re.compile(r"/(?:home|Users)/([^/\s\"']+)/|[A-Za-z]:\\Users\\[^\\]+\\")
URL = re.compile(r"https?://[^\s\"'<>`\\)]+")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
TOKEN = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b")


def inspect(path: str, data: bytes) -> set[str]:
    """Return policy violations without revealing matched values."""
    relative = Path(path)
    violations: set[str] = set()
    if relative.parts[0] in EXCLUDED_ROOTS:
        violations.add("excluded-tree")
    if ((relative.name.startswith(".env") and relative.name != ".env.example")
            or relative.suffix in {".pem", ".key", ".p12", ".pfx", ".bundle"}):
        violations.add("credential-or-backup-file")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        # The six reviewed previews need no text, EXIF or software metadata.
        offset = 8
        while offset + 12 <= len(data):
            length = int.from_bytes(data[offset:offset + 4], "big")
            kind = data[offset + 4:offset + 8]
            if kind in {b"tEXt", b"zTXt", b"iTXt", b"eXIf"}:
                violations.add("image-metadata")
            offset += length + 12
        return violations
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        violations.add("unreviewed-binary")
        return violations
    for match in PERSONAL_PATH.finditer(text):
        if match.group(1) != "example":  # Explicit synthetic security fixtures.
            violations.add("personal-absolute-path")
    if PRIVATE_KEY.search(text) or TOKEN.search(text):
        violations.add("credential-literal")
    if relative.suffix == '.py':
        # Audit annotations holding verbatim operator requests are not runtime
        # task prompts. Catch literal fields without printing the request text.
        try:
            tree = ast.parse(text)
        except SyntaxError:
            violations.add('invalid-python')
        else:
            private_fields = {'user_instruction', 'user_correction', 'operator_instruction'}
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if (isinstance(key, ast.Constant) and key.value in private_fields
                                and isinstance(value, ast.Constant)
                                and isinstance(value.value, str) and value.value.strip()):
                            violations.add('embedded-operator-request')
    for match in URL.finditer(text):
        try:
            url = urlsplit(match.group())
            host = url.hostname or ""
            if url.username or url.password:
                violations.add("url-userinfo")
            public = (host in PUBLIC_HOSTS or host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
                      or host in {"example.com", "example.org", "example.invalid"}
                      or host.endswith((".invalid", ".example", ".example.com", ".example.org")))
            if host and not public:
                violations.add("unreviewed-service-host")
        except ValueError:
            violations.add("invalid-url")
    # A literal LAN address is never necessary in a public configuration.
    for address in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", text):
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_private and not parsed.is_loopback and not parsed.is_unspecified:
            violations.add("private-ip-address")
    return violations


def inspect_history(repo: Path, ref: str, denylist: dict | None = None) -> dict:
    """Scan commit metadata and all reachable file versions, never just HEAD.

The optional denylist has `commits` and `literals` arrays. It must remain local:
its contents may themselves disclose removed infrastructure or history IDs.
"""
    denylist = denylist or {}
    denied = set(denylist.get('commits', []))
    literals = [value.encode().lower() for value in denylist.get('literals', []) if value]
    tip = subprocess.check_output(
        ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=repo, text=True,
    ).strip()
    commits = subprocess.check_output(['git', 'rev-list', tip], cwd=repo, text=True).splitlines()
    findings = []
    versions = set()
    for commit in commits:
        if commit in denied:
            findings.append({'object': commit, 'rules': ['retired-history-commit']})
        metadata = subprocess.check_output(
            ['git', 'show', '-s', '--format=%B%n%an <%ae>%n%cn <%ce>', commit], cwd=repo,
        )
        rules = inspect('commit-metadata.txt', metadata)
        if any(value in metadata.lower() for value in literals):
            rules.add('private-incident-identifier')
        if rules:
            findings.append({'object': commit, 'rules': sorted(rules)})
        entries = subprocess.check_output(['git', 'ls-tree', '-rz', commit], cwd=repo)
        for entry in entries.split(b'\0'):
            if not entry:
                continue
            fields, name = entry.split(b'\t', 1)
            mode, kind, oid = fields.decode().split()
            path = name.decode()
            if (path, oid) in versions:
                continue
            versions.add((path, oid))
            if mode == '120000' or kind != 'blob':
                findings.append({'path': path, 'rules': ['linked-object-requires-review']})
                continue
            data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=repo)
            rules = inspect(path, data)
            if any(value in data.lower() for value in literals):
                rules.add('private-incident-identifier')
            if any(value.encode() in data for value in denied):
                rules.add('retired-history-reference')
            if rules:
                findings.append({'path': path, 'rules': sorted(rules)})
    return {'commits': len(commits), 'file_versions': len(versions), 'findings': findings}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--staged", action="store_true")
    selection.add_argument('--ref', help='inspect this commit and all reachable history')
    parser.add_argument('--denylist', type=Path, help='private local incident identifiers (never commit)')
    args = parser.parse_args()
    if args.denylist and not args.ref:
        parser.error('--denylist requires --ref')
    if args.ref:
        denylist = json.loads(args.denylist.read_text()) if args.denylist else None
        report = inspect_history(ROOT, args.ref, denylist)
        print(json.dumps(report, indent=2))
        return int(bool(report['findings']))
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")[:-1]
    # Ignore rules alone do not protect already tracked or force-added files.
    result = subprocess.run(
        ['git', 'check-ignore', '--no-index', '-z', '--stdin'], cwd=ROOT,
        input='\0'.join(paths) + '\0', text=True, capture_output=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError('cannot verify publication allowlist')
    excluded = set(result.stdout.split('\0')) - {''}
    findings = []
    for path in paths:
        if args.staged:
            data = subprocess.check_output(["git", "show", ":" + path], cwd=ROOT)
        else:
            target = ROOT / path
            if target.is_symlink():
                findings.append({"path": path, "rules": ["symlink-requires-review"]})
                continue
            data = target.read_bytes()
        rules = inspect(path, data)
        if path in excluded:
            rules.add('ignored-file-is-tracked')
        if rules:
            findings.append({"path": path, "rules": sorted(rules)})
    print(json.dumps({"files": len(paths), "findings": findings}, indent=2))
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
