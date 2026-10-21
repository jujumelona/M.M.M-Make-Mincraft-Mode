"""Verify a shared Java file's validated, compositional materialization history."""
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence


def verify_java_patch_chain(receipts: Sequence[Mapping], final_hash: str) -> bool:
    """Every contribution must survive through patches to the exact final bytes.

    Whole-file replacement cannot vouch for an earlier contribution. Neither
    matching one latest receipt nor an unrelated checkpoint waives a broken
    intermediate link. Callers group only lexically admitted identical paths.
    """
    hashes = [row.get("after_sha256") for row in receipts]
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in hashes):
        return False
    # Walk backwards once; a large registry remains linear in its patch count.
    predecessors = defaultdict(set)
    for row in receipts:
        before = row.get("before_sha256")
        if (row.get("operation") == "java_patch" and isinstance(before, str)
                and re.fullmatch(r"[0-9a-f]{64}", before)):
            predecessors[row["after_sha256"]].add(before)
    reachable, pending = set(), [final_hash]
    while pending:
        value = pending.pop()
        if value in reachable:
            continue
        reachable.add(value)
        pending.extend(predecessors.get(value, ()))
    return all(value in reachable for value in hashes)
