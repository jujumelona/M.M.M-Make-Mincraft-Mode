from __future__ import annotations

"""Recognize compatible MiMo llama.cpp binaries without confusing build counters.

llama.cpp release version 0.6.0-dev reports build 1 despite containing
the newer MiMo tool-parser fix. The historical b11102 sequence is separate.
"""

import re

PINNED_MIMO_LLAMA_CPP_COMMIT = "d81235049384534c167caea52b85a694f6103d14"
MIN_LEGACY_MIMO_PARSER_BUILD = 11102


def mimo_server_version_compatible(
    reported: str,
    *,
    pinned_commit: str = PINNED_MIMO_LLAMA_CPP_COMMIT,
) -> tuple[bool, str]:
    """Allow the pinned source commit or an explicit legacy b11102+ build."""
    raw = reported.strip()
    commit = re.search(r"\bcommit\s+([0-9a-f]{8,40})\b", raw, re.I)
    if commit:
        short_sha = commit.group(1).lower()
        if pinned_commit.lower().startswith(short_sha):
            return True, "pinned-commit:" + short_sha
    match = re.search(r"\bb(\d{4,6})\b", raw, re.I)
    if not match:
        match = re.search(r"\bversion:\s*(\d{4,6})\b", raw, re.I)
    if match:
        build = int(match.group(1))
        if build >= MIN_LEGACY_MIMO_PARSER_BUILD:
            return True, "legacy-build:" + str(build)
        return False, "legacy build predates b11102: " + str(build)
    if commit:
        return False, "unrecognized or unpinned source commit: " + commit.group(1)
    return False, "no recognized compatible llama.cpp build or source commit"
