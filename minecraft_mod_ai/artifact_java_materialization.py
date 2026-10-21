"""Verify a shared Java file's validated, compositional materialization history."""
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


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


def checkpoint_java_patch_edge(checkpoint, root: Path, relative: str):
    """Take an edge only from the exact checkpoint's real same-path receipt."""
    receipt = checkpoint.get('receipt') or {}
    mat = receipt.get('materialization') or {}
    raw = mat.get('path')
    if (checkpoint.get('status') != 'SUCCESS' or receipt.get('status') != 'PASS'
            or checkpoint.get('target_path') != relative
            or mat.get('operation') != 'java_patch' or not isinstance(raw, str)
            or mat.get('before_sha256') != checkpoint.get('before_hash')
            or mat.get('after_sha256') != checkpoint.get('after_hash')):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    if path != root / relative:
        return None
    return {name: mat[name] for name in ('operation', 'before_sha256', 'after_sha256')}


def read_java_patch_history(root: Path, relative: str, final_hash: str) -> list[dict]:
    """Read durable host materialization proof across independent graph shards.

    This proves preserving byte updates, not behavior or runtime success. The
    graph still admits its own producer; final parsing and compilation remain
    mandatory. An external edit or whole-file overwrite has no preserving edge.
    """
    from .artifact_job_checkpoint import _read_record
    record = _read_record(root, 'paths', relative)
    if not isinstance(record, dict) or record.get('sha256') != final_hash:
        return []
    history = record.get('java_patch_history')
    if history is None:
        # Old notebooks already have exact per-job envelopes. Recover only
        # verified same-path edges, including an atomically published latest
        # checkpoint whose secondary jobs index was interrupted by a crash.
        history = []
        checkpoints = [record.get('job_checkpoint') or {}]
        for path in (root / '.mmm/artifact_jobs/jobs').glob('*.json'):
            import json
            key = json.loads(path.read_text(encoding='utf-8')).get('key')
            if not isinstance(key, str):
                raise ValueError('ARTIFACT_CHECKPOINT_CORRUPT')  # noqa: TRY004 - checkpoint API
            checkpoints.append(_read_record(root, 'jobs', key) or {})
        for checkpoint in checkpoints:
            edge = checkpoint_java_patch_edge(checkpoint, root, relative)
            if edge is not None and edge not in history:
                history.append(edge)
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError('ARTIFACT_CHECKPOINT_CORRUPT')
    # Legacy indexes may contain superseded jobs. Keep exactly the ancestry
    # of the current path receipt; orphaned edges cannot certify a caller.
    before_by_after = defaultdict(list)
    for row in history:
        if row.get('operation') == 'java_patch':
            before_by_after[row.get('after_sha256')].append(row)
    answer, seen, pending = [], set(), [final_hash]
    while pending:
        after = pending.pop()
        if after in seen:
            continue
        seen.add(after)
        for row in before_by_after.get(after, ()):
            if not isinstance(row.get('before_sha256'), str):
                raise ValueError('ARTIFACT_CHECKPOINT_CORRUPT')  # noqa: TRY004 - checkpoint API
            answer.append(row)
            pending.append(row['before_sha256'])
    return answer
