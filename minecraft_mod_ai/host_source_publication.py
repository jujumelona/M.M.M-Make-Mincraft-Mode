"""Transactional publication of host-owned gameplay occurrence producers.

The source model returns a Java owner and immutable plans. These plans are
published only during target compilation and are rolled back if any compiler,
source-provenance, native-mutation, or checkpoint gate fails. All publishers
share the same process and cross-process locks for the entire transaction.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


def _safe_path(root, relative):
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("HOST_SOURCE_PUBLICATION_PATH_ESCAPE")
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("HOST_SOURCE_PUBLICATION_SYMLINK")
    if candidate.exists() and not candidate.is_file():
        raise ValueError("HOST_SOURCE_PUBLICATION_NOT_FILE")
    return candidate


def _publication_paths(root, symbols, records):
    from .source_owned_custom_event import (plan_verified_custom_event, _obsolete_custom_sources)
    from .source_owned_ui_event import _obsolete_producers, _render_recipe

    manifest = _safe_path(root, "src/main/resources/fabric.mod.json")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    paths = {manifest}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("HOST_SOURCE_PUBLICATION_RECORD_INVALID")
        kind = record.get("kind")
        if kind == "custom":
            args = record.get("args")
            if not isinstance(args, dict) or "symbols" in args or "project_root" in args:
                raise ValueError("HOST_SOURCE_PUBLICATION_CUSTOM_INVALID")
            plan = plan_verified_custom_event(
                symbols=symbols, mod_id=metadata["id"], **args)
            paths.update(_safe_path(root, name) for name in plan["files"])
            digest = hashlib.sha256(
                (args["behavior_id"] + "\0" + args["source_site_ref"] + "\0"
                 + args.get("at", "HEAD")).encode()).hexdigest()[:16]
            paths.add(_safe_path(root, ".mmm/source-producers/" + digest + ".json"))
            for receipt_path, _, previous in _obsolete_custom_sources(
                    root, symbols, metadata, args["behavior_id"],
                    args["package"], digest):
                paths.add(_safe_path(root, receipt_path.relative_to(root)))
                paths.update(_safe_path(root, name) for name in previous["files"])
        elif kind == "ui":
            plan = record.get("plan")
            if not isinstance(plan, dict) or plan != _render_recipe(plan["recipe"], symbols):
                raise ValueError("HOST_SOURCE_PUBLICATION_UI_PLAN_INVALID")
            paths.update(_safe_path(root, name) for name in plan["files"])
            paths.add(_safe_path(root, ".mmm/typed-ui-producers/" + plan["digest"] + ".json"))
            for receipt, obsolete in _obsolete_producers(root, plan, symbols):
                paths.add(_safe_path(root, str(receipt.relative_to(root))))
                paths.update(_safe_path(root, name) for name in obsolete["files"])
        else:
            raise ValueError("HOST_SOURCE_PUBLICATION_KIND_INVALID")
    return sorted(paths)


@contextmanager
def source_publication_transaction(*, project_root, symbols, records, tracked_paths=()):
    """Publish planned producers, then require the caller's COMPLETE verification.

    The transaction completes only if its with-block returns successfully.
    Rejected javac repairs, omitted source evidence, cancelled model calls,
    and failed checkpoint persistence all roll back every producer file.
    """
    if not records and not tracked_paths:
        yield
        return

    from contextlib import nullcontext
    from .project_write_lock import project_write_lock
    from .source_owned_custom_event import (
        _atomic_write, _producer_cross_process_lock, materialize_verified_custom_event,
    )
    from .source_owned_ui_event import materialize_typed_ui_occurrence

    root = Path(project_root).resolve()
    producer_lock = (_producer_cross_process_lock(root / ".mmm/source-producer.lock")
                     if records else nullcontext())
    with project_write_lock(root), producer_lock:
        paths = sorted(
            (set(_publication_paths(root, symbols, records)) if records else set())
            | {_safe_path(root, path) for path in tracked_paths}
        )
        before = {path: path.read_bytes() if path.exists() else None for path in paths}
        try:
            for record in records:
                if record["kind"] == "custom":
                    materialize_verified_custom_event(
                        project_root=root, symbols=symbols, **record["args"])
                else:
                    materialize_typed_ui_occurrence(
                        record["plan"], project_root=root, symbols=symbols)
            yield
        except BaseException:
            failures = []
            for path in reversed(paths):
                original = before[path]
                try:
                    if original is None:
                        if path.is_file():
                            path.unlink()
                    elif not path.is_file() or path.read_bytes() != original:
                        _atomic_write(path, original.decode("utf-8"))
                except (OSError, UnicodeDecodeError, ValueError) as exc:
                    failures.append(str(path) + ": " + str(exc))
            if failures:
                raise RuntimeError(
                    "HOST_SOURCE_PUBLICATION_ROLLBACK_INCOMPLETE: "
                    + "; ".join(failures)) from None
            raise
