"""Which code produced a verdict — resolved fail-closed, never guessed.

A verdict row that cannot say which code wrote it is the defect issue #13
exists to close, so this module refuses to produce a value it cannot stand
behind. `None` is a legal `code_version` only on the READ path, where it means
"a row written before provenance existed". It is never produced here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SHA_CHARS = 12
TREE_DIGEST_CHARS = 12


class GitUnavailable(RuntimeError):
    """The repository state could not be identified at all."""


class DirtyTree(RuntimeError):
    """The working tree has uncommitted changes and --allow-dirty was not given."""


def code_version(*, head: str | None, tree_diff: str, allow_dirty: bool) -> str:
    """Identify the code behind a run. Pure; `head`/`tree_diff` come from git.

    An empty `tree_diff` means a clean tree, and the short sha alone identifies
    the code. A dirty tree is refused unless explicitly allowed, and when it is
    allowed the diff is HASHED rather than flagged: a bare `-dirty` marker
    collapses every distinct uncommitted tree at one commit into a single value,
    so edit -> run -> edit -> resume would append incompatible rows under a
    version string that never changed.
    """
    if not head:
        raise GitUnavailable(
            "cannot identify the code version: not a git repository, or git is unavailable. "
            "Verdict rows must record which code produced them; refusing to start."
        )
    if not tree_diff:
        return head[:SHA_CHARS]
    if not allow_dirty:
        raise DirtyTree(
            f"working tree has uncommitted changes at {head[:SHA_CHARS]}. A calibration sweep "
            "should run from a committed tree so its verdicts can be reproduced. "
            "Commit the changes, or pass --allow-dirty to record a content-hashed version."
        )
    digest = hashlib.sha256(tree_diff.encode()).hexdigest()[:TREE_DIGEST_CHARS]
    return f"{head[:SHA_CHARS]}-dirty-{digest}"


@dataclass(frozen=True)
class ProvenanceAudit:
    """What one results file says about where its rows came from."""

    code_versions: tuple[str, ...]
    config_digests: tuple[str, ...]
    base_urls: tuple[str, ...]
    rows_without_provenance: int
    malformed_rows: int
    total_rows: int

    @property
    def is_mixed(self) -> bool:
        """True if this file cannot be shown to be one coherent run.

        Two known values of anything is a mixture. So is one known value beside
        rows that predate provenance — treating unknown as "probably the same"
        is what let the #10 near-miss look clean. A row that will not parse
        counts the same way: its provenance is unreadable, not absent, so a
        damaged file must never come back looking coherent.

        A file that is UNIFORMLY unknown is merely unauditable, which
        `rows_without_provenance` reports on its own.
        """
        distinct = max(len(self.code_versions), len(self.config_digests), len(self.base_urls))
        if distinct > 1:
            return True
        if self.malformed_rows > 0 and distinct > 0:
            return True
        return distinct == 1 and self.rows_without_provenance > 0


def audit_results_file(path: Path | str) -> ProvenanceAudit:
    """Collect the distinct provenance across a results file, after the fact.

    Reads raw JSON rather than Verdict: an audit must still report on a file
    whose rows no longer satisfy the current schema, since that is exactly the
    kind of file worth auditing.
    """
    seen: dict[str, set[str]] = {"code_version": set(), "config_digest": set(), "base_url": set()}
    unknown = 0
    malformed = 0
    total = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
        except ValueError:
            # Counted, never skipped silently: an unreadable row is provenance
            # we cannot check, and dropping it would let a damaged file report
            # as coherent.
            malformed += 1
            continue
        for field, values in seen.items():
            if (value := record.get(field)) is not None:
                values.add(value)
        if record.get("code_version") is None and record.get("base_url") is None:
            unknown += 1
    return ProvenanceAudit(
        code_versions=tuple(sorted(seen["code_version"])),
        config_digests=tuple(sorted(seen["config_digest"])),
        base_urls=tuple(sorted(seen["base_url"])),
        rows_without_provenance=unknown,
        malformed_rows=malformed,
        total_rows=total,
    )


def _git(*args: str) -> str | None:
    """Run a git command, or None if git or the repository is unavailable."""
    try:
        done = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    return done.stdout if done.returncode == 0 else None


def read_tree_diff() -> str:
    """Everything uncommitted that could change a judgment.

    Tracked modifications plus the contents of untracked, non-ignored Python
    files — an untracked module can be imported and change behaviour just as a
    tracked edit can, so listing only its name would not distinguish two
    different versions of it.
    """
    diff = _git("diff", "HEAD") or ""
    listed = _git("ls-files", "--others", "--exclude-standard", "*.py") or ""
    parts = [diff]
    for name in sorted(filter(None, listed.splitlines())):
        try:
            body = Path(name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            body = ""
        parts.append(f"--- untracked {name}\n{body}")
    return "".join(parts)


def resolve_code_version(*, allow_dirty: bool = False) -> str:
    """`code_version` for the repository this process is running from."""
    head = (_git("rev-parse", "HEAD") or "").strip() or None
    return code_version(head=head, tree_diff=read_tree_diff(), allow_dirty=allow_dirty)
