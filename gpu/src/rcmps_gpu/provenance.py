"""Provenance records of the imported sources, and their check.

The code was imported byte for byte from the research tree. optimizer/provenance.json (the
optimizer runtime) and excitations/provenance.json (the excitation solver) record the
sha256 of each file as imported and every edit made since, one entry per commit in commit
order. An entry has a "commit_subject", optionally a "change" (text), and any of:

- "imports": {file: [[n, old, new], ...]}: line n of the file after the commit is `new`
  and was `old`, and the edit only makes a flat sibling import package-relative
  (old == flat_import(new), checked);
- "hunks": {file: [[n, [old lines], [new lines]], ...]}: the commit replaced `old lines`
  by `new lines`, which start at line n of the file after the commit (for a deletion,
  `new lines` is empty and n is the line the old ones preceded);
- "replaced": {file: {"before": sha256, "after": sha256}}: the commit rewrote the file,
  whose diff is in git; earlier edits of that file are covered by the sha256 `before` only;
- "added": {file: sha256}: the commit added the file with that sha256;
- "removed": {file: sha256}: the commit deleted the file, whose sha256 as imported (or as
  added) this is; its edits before the deletion are not checked against any content.

Lines are written without their newline ("\\n") and line numbers start at 1. The line
edits of one file in one entry ("imports" and "hunks" together) do not overlap; a file
that an entry adds, removes or replaces has no other edit in that entry. verify() reverts
the entries in memory, newest first, working on bytes, and requires the imported sha256
of every file.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

KINDS = ("imports", "hunks", "replaced", "added", "removed")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def flat_import(line):
    """The research form of a line that the packaging commit made package-relative."""
    line = re.sub(r"^(\s*)from \. import ", r"\1import ", line, count=1)
    line = re.sub(r"^(\s*)from \.(?=\w)", r"\1from ", line, count=1)
    return re.sub(r"__import__\('(\w+)',globals\(\),level=1\)", r"__import__('\1')", line)


def _lines(texts):
    return [(text + "\n").encode() for text in texts]


def _revert(raw, edits, name, subject):
    """Undo the line edits of one file made by one commit (positions in the edited file)."""
    lines = raw.splitlines(keepends=True)
    # A deletion at line n (no new lines) comes before an edit of line n itself.
    edits = sorted(edits, key=lambda edit: (edit[0], len(edit[2])))
    for (n, _, new), (m, _, later) in zip(edits, edits[1:]):
        if n + len(new) > m or n == m and not new and not later:
            raise ValueError(f"{subject}: overlapping edits of {name} at lines {n} and {m}")
    for n, old, new in reversed(edits):
        if n < 1 or lines[n - 1:n - 1 + len(new)] != _lines(new):
            raise ValueError(f"{name} differs from the recorded edit of line {n} ({subject})")
        lines[n - 1:n - 1 + len(new)] = _lines(old)
    return b"".join(lines)


def verify(directory, imported, edits, names):
    """Check that the files `names` of `directory`, with the recorded `edits` reverted, are
    the imported files: `imported` maps each imported file name to its sha256. Raises
    ValueError on the first difference; returns the names of the files edited or added since."""
    directory = Path(directory)
    names = set(names)
    files = {name: (directory / name).read_bytes() for name in names}   # bytes, or a sha256 str
    removed = {}   # deleted files: sha256 as imported or added
    edited = set()
    for entry in reversed(edits):
        subject = entry["commit_subject"]
        unknown = set(entry) - set(KINDS) - {"commit_subject", "change"}
        if unknown:
            raise ValueError(f"{subject}: unknown edit kinds {sorted(unknown)}")
        touched = [name for kind in ("added", "removed", "replaced") for name in entry.get(kind, {})]
        touched += set(entry.get("imports", {})) | set(entry.get("hunks", {}))
        if len(touched) != len(set(touched)):
            raise ValueError(f"{subject}: a file is both edited by lines and added, removed or replaced")
        for name, digest in entry.get("added", {}).items():
            content = removed.pop(name) if name in removed else files.pop(name, None)
            if content is None or _digest(content) != digest:
                raise ValueError(f"{name} differs from the file added by '{subject}'")
        for name, digest in entry.get("removed", {}).items():
            if name in files or name in removed:
                raise ValueError(f"{name} was removed by '{subject}' but exists")
            removed[name] = digest
        for name, change in entry.get("replaced", {}).items():
            if name in removed:
                continue
            if name not in files or _digest(files[name]) != change["after"]:
                raise ValueError(f"{name} differs from the file written by '{subject}'")
            files[name] = change["before"]
        lines = {}
        for name, rows in entry.get("imports", {}).items():
            for n, old, new in rows:
                if old == new or flat_import(new) != old:
                    raise ValueError(f"{subject}: the edit of {name} line {n} is not a relative import")
                lines.setdefault(name, []).append((n, [old], [new]))
        for name, rows in entry.get("hunks", {}).items():
            lines.setdefault(name, []).extend((n, old, new) for n, old, new in rows)
        for name, rows in lines.items():
            if name in removed:
                continue
            if name not in files or isinstance(files[name], str):
                raise ValueError(f"{subject}: no file {name} to revert")
            files[name] = _revert(files[name], rows, name, subject)
        edited.update(touched)
    files.update(removed)
    if set(files) != set(imported):
        raise ValueError(f"files {sorted(set(files) ^ set(imported))} are not accounted for by the record")
    for name, content in sorted(files.items()):
        if _digest(content) != imported[name]:
            raise ValueError(f"{name} with its recorded edits reverted is not the imported file")
    return sorted(edited & names)


def _digest(content):
    return content if isinstance(content, str) else sha256(content)


def inert(path):
    """True if the module at `path` holds only a docstring and assignments of constants."""
    tree = ast.parse(Path(path).read_bytes())
    return all(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
               or isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
               and all(isinstance(target, ast.Name) for target in node.targets)
               for node in tree.body)
