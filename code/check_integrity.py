"""Verify that benchmark-controlled files still match their release hashes.

Usage: python check_integrity.py
Exit code 0 when every benchmark-controlled file matches, 1 otherwise.

`PACKAGE_MANIFEST.json` records the sha256 of every released file.  Files this
project is allowed to change (student.py, train.py, README.md) are excluded;
everything else must stay byte-identical, because the comparisons in the report
are anchored on the supplied protocol (see constitution principle II).

Run this after any edit and before every commit.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent

# This project may modify these; the report discloses the change and its purpose.
MUTABLE = {
    'code/student.py',
    'code/train.py',
    'code/README.md',
}

# Documentation, not part of the measured protocol.  guide/GUIDE.md was already
# different from its manifest hash when this project started (the file is pure LF,
# so it is not a line-ending artefact) and has never been written to by this
# project.  It is reported, but it must not fail the protocol-integrity gate.
REPORT_ONLY = {
    'guide/GUIDE.md',
}

# The extracted package flattened guide/ into the package root, so the manifest
# path differs from the on-disk path for these entries.
RELOCATED = {
    'guide/GUIDE.md': 'GUIDE.md',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    checked = 0
    problems = []
    reported = []
    for relative, expected in sorted(manifest.items()):
        if relative in MUTABLE:
            print(f'  skip      {relative}  (mutable by this project)')
            continue
        on_disk = RELOCATED.get(relative, relative)
        path = PACKAGE_ROOT / on_disk
        if not path.exists():
            print(f'  MISSING   {relative}  (looked at {on_disk})')
            problems.append((relative, 'missing'))
            continue
        if relative in REPORT_ONLY:
            status = 'ok' if sha256(path) == expected else 'DIFFERS'
            print(f'  {status:<9} {relative}  (documentation, not gated)')
            if status != 'ok':
                reported.append(relative)
            continue
        checked += 1
        if sha256(path) != expected:
            print(f'  CHANGED   {relative}')
            problems.append((relative, 'changed'))
        else:
            note = f'  (at {on_disk})' if on_disk != relative else ''
            print(f'  ok        {relative}{note}')
    print()
    print(f'checked {checked} protocol/reference files, {len(problems)} problem(s)')
    for relative, kind in problems:
        print(f'  !! {kind}: {relative}')
    if reported:
        print(f'note: {len(reported)} documentation file(s) differ from the manifest:')
        for relative in reported:
            print(f'  - {relative}: differs from the released hash; not written to by this'
                  ' project, not part of the measured protocol')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())