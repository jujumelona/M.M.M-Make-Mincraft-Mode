"""Checkpointed one-method-at-a-time Java authoring for independent features.

The direct path is deliberately narrow: it never claims an external donor was
installed, never replaces a persistent-state plan, and never certifies gameplay.
The existing typed-IR implementation continues to own those cases.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .construction_retrieval import ConstructionEvidenceIndex
from .first_pass_java_declarations import declaration_frame, render_declaration
from .java_region_parser import _parse


_CLAIM_MUTEX = threading.Lock()
_CLAIM_LOCKS = {}


@contextmanager
def _claim_checkpoint(path):
    """One notebook/kernel owns inference and publication for this feature."""
    from .runner_lock_contract import _acquire, _release

    lock_path = path.with_suffix(".lock")
    with _CLAIM_MUTEX:
        mutex = _CLAIM_LOCKS.setdefault(str(lock_path), threading.RLock())
    with mutex:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        deadline = time.monotonic() + 1800
        try:
            while not acquired:
                try:
                    _acquire(descriptor)
                    acquired = True
                except (BlockingIOError, OSError) as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CONSTRUCTION_DIRECT_CHECKPOINT_LOCK_TIMEOUT") from error
                    time.sleep(0.2)
            yield
        finally:
            if acquired:
                _release(descriptor)
            os.close(descriptor)


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")).hexdigest()


def eligible(bindings, context, behaviors):
    """Never bypass donor integration, retained typed effects or state ownership."""
    if not bindings.get("request_feature_entrypoint") or not behaviors:
        return False
    if context.get("donor_callable_symbols") or context.get("reference_port_obligations"):
        return False
    if any(behavior.get("state_reads") or behavior.get("state_writes")
           or behavior.get("source_status", "grounded") != "grounded"
           for behavior in behaviors):
        return False
    if json.loads(bindings.get("owned_runtime_records_json") or "[]"):
        return False
    retained = json.loads(bindings.get("retained_typed_behavior_json") or "{}")
    if any(retained.get(key) for key in ("functions", "event_bindings", "ui_owned_records")):
        return False
    return True


def _read_checkpoint(path, identity):
    if not path.is_file():
        return {}
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("identity") != identity:
        raise ValueError("CONSTRUCTION_DIRECT_CHECKPOINT_BINDING_MISMATCH")
    records = envelope.get("members")
    if not isinstance(records, dict) or envelope.get("digest") != _digest(records):
        raise ValueError("CONSTRUCTION_DIRECT_CHECKPOINT_CORRUPT")
    for value in records.values():
        if not isinstance(value, str) or not value:
            raise ValueError("CONSTRUCTION_DIRECT_CHECKPOINT_MEMBER_INVALID")
    return records


def _save_checkpoint(path, identity, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema_version": "mmm/construction-direct-v1",
                "identity": identity, "members": members,
                "digest": _digest(members)}
    temp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _evidence(index, intent, source_pages):
    """Expose only ranked, inspected target symbols and actual installed source."""
    goal = index.prepare(intent, kinds=("METHOD", "CONSTRUCTOR", "FIELD", "CLASS"))
    page = index.page(goal, 0)
    result = {"target_api": page["matches"][:8], "source": []}
    for candidate in page["source_candidates"][:4]:
        ref, offset = candidate["source_ref"], candidate["offset"]
        units = source_pages.get(ref, ())
        if not units or offset >= len(units):
            continue
        unit = units[offset]
        if unit.get("authority") in {
            "reference_only_port_required", "reference_only_requires_target_adaptation"
        }:
            continue
        result["source"].append({
            "path": ref, "authority": unit.get("authority", "inspected_materialized_source"),
            "signature": unit.get("signature", ""),
            "source_sha256": unit.get("source_sha256", ""),
            "code": unit.get("code", "")[:8000],
        })
    return result


def compose_feature(router, spec, context, behaviors, *, package, name, owner):
    """Write independently owned complete Java methods and preserve each accepted one."""
    root = Path(context["project_root"]).resolve()
    identity = _digest({
        "spec": spec, "owner": owner, "behaviors": behaviors,
        "target": context["target"],
        "namespace": context.get("namespace", ""),
        "prerequisites": context["prerequisite_hashes"],
        "contract": context.get("authoring_contract", {}),
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    checkpoint = root / ".mmm" / "construction_direct" / (identity + ".json")
    with _claim_checkpoint(checkpoint):
        return _compose_claimed(router, spec, context, behaviors, package, name,
                                owner, root, checkpoint, identity)


def _compose_claimed(router, spec, context, behaviors, package, name, owner,
                     root, checkpoint, identity):
    saved = _read_checkpoint(checkpoint, identity)
    expected = {str(row["behavior_id"]) for row in behaviors}
    if set(saved) - expected:
        raise ValueError("CONSTRUCTION_DIRECT_CHECKPOINT_UNKNOWN_MEMBER")
    # Never accept duplicate logical behaviors under different dictionary keys.
    if len(expected) != len(behaviors):
        raise ValueError("CONSTRUCTION_DIRECT_BEHAVIOR_DUPLICATE")
    symbols = dict(context["symbols"])
    sources = context.get("source_pages", {})
    index = ConstructionEvidenceIndex(symbols, sources, router,
        target_path=spec["target_path"], owner=owner,
        authoring_contract=context.get("authoring_contract", {}))
    metadata = {"visibility": "private", "static": False, "final": False,
                "return_type": "void", "parameters": [], "throws": [],
                "annotations": [], "type_parameters": ""}
    completed = []
    for behavior in behaviors:
        behavior_id = str(behavior["behavior_id"])
        method = "registerBehavior" + hashlib.sha256(
            behavior_id.encode("utf-8")).hexdigest()[:20]
        frame = declaration_frame("method", method, metadata)
        evidence = _evidence(index, " ".join((
            str(behavior["intent"]), str(behavior.get("effects", "")),
            str(behavior.get("trigger", "")))), sources)
        task = {
            "owner": owner, "method": method, "original_source": behavior["source_quote"],
            "intent": behavior["intent"], "trigger": behavior.get("trigger", ""),
            "effects": behavior["effects"], "target_api_and_source": evidence,
            "authoring_contract": context.get("authoring_contract", {}),
            # Preserve the full accepted work in the checkpoint; send a
            # repository-map-like compact member catalogue to the small model.
            "accepted_member_catalogue": [
                {"method": item["method"], "source_sha256": hashlib.sha256(
                    item["source"].encode("utf-8")).hexdigest()}
                for item in completed],
            "previous_relevant_member_sources": completed[-2:],
            "response_source_frame": frame,
        }
        messages = [
            {"role": "system", "content": (
                "Implement exactly one independently owned Fabric 26.2 behavior "
                "as COMPLETE Java method source. The host fixes the signature; "
                "return the entire declaration with exactly that signature, not "
                "a planning decision or a fragment. Use real inspected APIs "
                "from target_api_and_source and exact prerequisite code, not "
                "imagined imports or APIs. Do not add another member, package, "
                "class or unrelated effects. The resulting method must implement "
                "the original feature, not TODO, empty, or unsupported code. "
                "Source evidence marked rendered_sibling_not_yet_target_compiled "
                "is local work, not proof of API availability.")
            },
            {"role": "user", "content": json.dumps(task, ensure_ascii=False, default=str)},
        ]
        member = None
        # Previously accepted complete members are resumed exactly. For one
        # invalid candidate, allow one diagnostic-guided correction, never an
        # unbounded planner or sampler loop.
        attempts = 1 if behavior_id in saved else 2
        for attempt in range(attempts):
            candidate = (saved[behavior_id] if behavior_id in saved else
                         router.generate_java_source(
                             "coder", messages,
                             data_name="construction_direct_" + behavior_id[:60],
                             source_frame=frame))
            try:
                if (not isinstance(candidate, str)
                        or not candidate.startswith(frame["prefix"])
                        or not candidate.endswith(frame["suffix"])):
                    raise ValueError("CONSTRUCTION_DIRECT_SIGNATURE_INVALID")
                body = candidate[len(frame["prefix"]):-len(frame["suffix"])].strip()
                if not body or body in {"return;", ";"} or re.search(
                        r"\b(?:TODO|FIXME|UnsupportedOperationException)\b", body):
                    raise ValueError("CONSTRUCTION_DIRECT_BODY_INCOMPLETE")
                member = render_declaration("method", method,
                                            {**metadata, "source": candidate}, owner=name)
                break
            except ValueError as error:
                if behavior_id in saved or attempt + 1 >= attempts:
                    raise
                messages.append({"role": "user", "content": json.dumps({
                    "repair_current_declaration": True,
                    "error": str(error)[:1600],
                    "previous_candidate": str(candidate)[:8000],
                    "required_signature": frame["prefix"],
                    "instruction": "Return the whole corrected method, keep the exact frame.",
                }, ensure_ascii=False)})
        if member is None:
            raise ValueError("CONSTRUCTION_DIRECT_BODY_NOT_VALIDATED")
        if behavior_id not in saved:
            saved[behavior_id] = member
            _save_checkpoint(checkpoint, identity, saved)
        completed.append({"method": method, "source": member})
        # Source evidence is retrievable by later independent behaviors, but
        # never promoted to a compiled/installable API declaration.
        ref = "direct-accepted:" + hashlib.sha256(behavior_id.encode()).hexdigest()[:20]
        index.include_source(ref, [{
            "kind": "accepted_java_member", "owner": owner,
            "signature": method + "()", "code": member,
            "source_sha256": hashlib.sha256(member.encode("utf-8")).hexdigest(),
            "authority": "rendered_sibling_not_yet_target_compiled",
        }])
        sources[ref] = [{
            "signature": method + "()", "code": member,
            "source_sha256": hashlib.sha256(member.encode("utf-8")).hexdigest(),
            "authority": "rendered_sibling_not_yet_target_compiled",
        }]
    lifecycle = "\n".join(
        "this." + row["method"] + "();" for row in completed
    )
    members = "\n\n".join(row["source"] for row in completed)
    source = ("package " + package + ";\n\npublic final class " + name
              + " implements net.fabricmc.api.ModInitializer {\n"
              + members + "\n@Override\npublic void onInitialize() {\n"
              + lifecycle + "\n}\n}\n")
    _parse(source)
    if context.get("verify_source"):
        context["verify_source"](source, spec["target_path"])
    from .construction_coverage import record_authored_source_calls
    record_authored_source_calls(
        root, spec["target_path"], source, selected_ports=[],
        installed_donor_calls=[], source_kind="direct")
    return source
