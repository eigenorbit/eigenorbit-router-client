"""Vendor the router's contract v1 schemas and examples into tests/contract/.

The offline parity suite (tests/test_offline_parity.py) runs against the
router's own JSON schemas and examples. They are copied here byte-for-byte
from a router commit and recorded in tests/contract/SOURCE.json with the
commit and one sha256 per file. Only this script writes that tree; never edit
it by hand.

Modes:

    python scripts/sync_contract.py --router PATH [--ref REF]
        Copy contract/v1/schemas/*.schema.json and contract/v1/examples/*.json
        from the router checkout at PATH, as committed at REF (default HEAD),
        into tests/contract/v1/, delete vendored files the router's contract
        at REF does not have, and rewrite SOURCE.json. CONTRACT.md is not
        vendored.

    python scripts/sync_contract.py --check --router PATH [--ref REF]
        Exit 1 and list every added, removed or changed file when the router's
        contract at REF differs from the vendored copy.

    python scripts/sync_contract.py --verify
        Exit 1 when a vendored file's sha256 differs from SOURCE.json, or a
        file is missing or unlisted. Needs no router checkout (CI runs this).

`--root` points at another client checkout (the tests use a temporary copy).
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROUTER_REPO = "eigenorbit-router"
CONTRACT_PATH = "contract/v1"
# Vendored subdirectory -> the filename suffix vendored from it.
KINDS = {"schemas": ".schema.json", "examples": ".json"}
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = Path("tests") / "contract"


def _v1_dir(root: Path) -> Path:
    return root / VENDOR_DIR / "v1"


def _source_json(root: Path) -> Path:
    return root / VENDOR_DIR / "SOURCE.json"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(router: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(router), *args], capture_output=True, check=False
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {router}: {detail}")
    return completed.stdout


def resolve_commit(router: Path, ref: str) -> str:
    """The full sha of `ref` in the router checkout."""
    return _git(router, "rev-parse", "--verify", f"{ref}^{{commit}}").decode().strip()


def router_files(router: Path, ref: str) -> dict[str, bytes]:
    """The router's vendorable contract files at `ref`: `kind/name` -> bytes."""
    files: dict[str, bytes] = {}
    for kind, suffix in KINDS.items():
        prefix = f"{CONTRACT_PATH}/{kind}/"
        listing = _git(router, "ls-tree", "--name-only", ref, "--", prefix)
        for path in listing.decode().splitlines():
            name = path[len(prefix) :]
            if not path.startswith(prefix) or "/" in name or not name.endswith(suffix):
                continue
            files[f"{kind}/{name}"] = _git(router, "show", f"{ref}:{path}")
    return files


def vendored_files(root: Path) -> dict[str, bytes]:
    """Every file under tests/contract/v1: `kind/name` -> bytes."""
    v1 = _v1_dir(root)
    if not v1.is_dir():
        return {}
    return {
        path.relative_to(v1).as_posix(): path.read_bytes()
        for path in sorted(v1.rglob("*"))
        if path.is_file()
    }


def verify(root: Path) -> list[str]:
    """Problems between the vendored tree and SOURCE.json; empty when clean."""
    source_path = _source_json(root)
    if not source_path.is_file():
        return [f"{source_path.relative_to(root)} is missing"]
    try:
        listed = json.loads(source_path.read_text())["files"]
    except (ValueError, KeyError, TypeError):
        return [f"{source_path.relative_to(root)} has no files map"]
    present = vendored_files(root)
    problems = []
    for key in sorted(set(listed) | set(present)):
        if key not in present:
            problems.append(f"missing: {key}")
        elif key not in listed:
            problems.append(f"unlisted: {key}")
        elif _sha256(present[key]) != listed[key]:
            problems.append(f"changed: {key}")
    return problems


def diff_against_router(root: Path, router: Path, ref: str = "HEAD") -> list[str]:
    """Differences between the vendored tree and the router's contract at `ref`."""
    theirs = router_files(Path(router), ref)
    ours = vendored_files(root)
    problems = []
    for key in sorted(set(theirs) | set(ours)):
        if key not in ours:
            problems.append(f"added: {key}")
        elif key not in theirs:
            problems.append(f"removed: {key}")
        elif theirs[key] != ours[key]:
            problems.append(f"changed: {key}")
    return problems


def vendor(root: Path, router: Path, ref: str = "HEAD") -> str:
    """Copy the router's contract at `ref` into the tree; return its commit."""
    commit = resolve_commit(router, ref)
    files = router_files(router, commit)
    if not files:
        raise RuntimeError(f"{router} has no {CONTRACT_PATH} schemas or examples at {ref}")
    v1 = _v1_dir(root)
    for key in vendored_files(root):
        if key not in files:
            (v1 / key).unlink()
    for key, content in files.items():
        target = v1 / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    source = {
        "commit": commit,
        "files": {key: _sha256(content) for key, content in sorted(files.items())},
        "path": CONTRACT_PATH,
        "repo": ROUTER_REPO,
    }
    _source_json(root).write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    return commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Vendor, check or verify the router contract v1 fixtures."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="compare with the router; do not write"
    )
    mode.add_argument(
        "--verify", action="store_true", help="check the tree against SOURCE.json"
    )
    parser.add_argument("--router", type=Path, help="path to a router checkout")
    parser.add_argument("--ref", default="HEAD", help="router commit or ref (default HEAD)")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="client checkout (default: this one)"
    )
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify(args.root)
        for problem in problems:
            print(problem)
        if problems:
            print("the vendored contract differs from tests/contract/SOURCE.json; "
                  "re-run scripts/sync_contract.py --router <router checkout>")
            return 1
        print("vendored contract matches tests/contract/SOURCE.json")
        return 0

    if args.router is None:
        parser.error("--router is required unless --verify is given")
    try:
        if args.check:
            problems = diff_against_router(args.root, args.router, args.ref)
            for problem in problems:
                print(problem)
            if problems:
                print(f"the router contract at {args.ref} differs from the vendored copy")
                return 1
            print(f"vendored contract matches the router at {args.ref}")
            return 0
        commit = vendor(args.root, args.router, args.ref)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"vendored {CONTRACT_PATH} from {ROUTER_REPO}@{commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
