"""Checkpointed one-method-at-a-time Java authoring for independent features.

The direct path never replaces a persistent-state plan or certifies gameplay.
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
from .model_concurrency import model_execution_publication

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
    """Never bypass retained typed effects or state ownership."""
    if not bindings.get("request_feature_entrypoint") or not behaviors:
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
        with model_execution_publication():
            temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _evidence(index, intent, source_pages):
    """Expose only ranked, inspected target symbols and actual installed source."""
    goal = index.prepare(intent, kinds=("METHOD", "CONSTRUCTOR", "FIELD", "CLASS"))
    page = index.page(goal, 0)
    # Direct Java source bypassed the typed operation binder, allowing the 9B
    # model to pick FabricEntityDataRegistry or EntityRenderers for entity
    # *type* registration. Apply the same functional admission gate before
    # a candidate is ever shown to the direct source generator.
    from .construction_api_admission import api_semantic_admissible, root_action_stems
    action_required = bool(root_action_stems(intent))
    target_api = []
    rejected = 0
    # The Minecraft gameplay author must not receive Vulkan/graphics SDK
    # symbols merely because an intent mentions "space" or "world".
    # Real client rendering tasks are the exception, never server gameplay.
    path_lower = str(getattr(index, "target_path", "")).casefold()
    client_rendering = ("render" in path_lower or "src/client/" in path_lower)
    for candidate in page["matches"]:
        row = index.symbols.get(candidate.get("symbol_id"), {})
        owner_name = str(row.get("owner", ""))
        if owner_name.startswith(("org.lwjgl.", "org.joml.")) and not client_rendering:
            rejected += 1
            continue
        if (row.get("kind") == "METHOD"
                and row.get("owner") == getattr(index, "owner", None)):
            # A planned declaration of the very class being authored is not
            # a target API implementation. Accepted siblings remain available
            # as actual source below, rather than inviting recursive self-calls.
            rejected += 1
            continue
        if (action_required and row.get("kind") == "METHOD"
                and not api_semantic_admissible("call", intent, row)):
            rejected += 1
            continue
        if (not str(getattr(index, "target_path", "")).startswith("src/client/")
                and row.get("side") == "CLIENT"):
            rejected += 1
            continue
        target_api.append(candidate)
        if len(target_api) >= 8:
            break
    # A ranked free-text search can return only irrelevant classes. For
    # recognizable gameplay kinds, recover a small set of *actual indexed*
    # official/Fabric symbols instead of sending ungrounded suggestions.
    hints = (intent + " " + path_lower).casefold()
    anchors = {
        "entity": ("net.minecraft.world.entity.EntityType",
                   "net.minecraft.world.entity.MobCategory",
                   "net.minecraft.core.Registry",
                   "net.minecraft.core.registries.BuiltInRegistries"),
        "block": ("net.minecraft.world.level.block.Block",
                  "net.minecraft.world.level.block.state.BlockBehaviour",
                  "net.minecraft.core.registries.BuiltInRegistries"),
        "item": ("net.minecraft.world.item.Item",
                 "net.minecraft.core.registries.BuiltInRegistries"),
        "command": ("net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback",
                    "net.minecraft.commands.Commands"),
    }
    from .construction_state import symbol_record
    known_keys = {item.get("symbol_id") for item in target_api}
    for cue, owners in anchors.items():
        if cue not in hints:
            continue
        for owner_name in owners:
            # Never fabricate a symbol from a guessed API name: it must exist
            # as a CLASS record in the exact selected target symbol index.
            for key in index.by_owner.get(owner_name, ()):
                row = index.symbols.get(key, {})
                if row.get("kind") != "CLASS" or key in known_keys:
                    continue
                target_api.append(symbol_record(key, row))
                known_keys.add(key)
                break
            if len(target_api) >= 8:
                break
        if len(target_api) >= 8:
            break
    result = {"target_api": target_api, "source": [],
              "typed_input_producers": [], "rejected_unrelated_api_count": rejected}
    # A correct registration API is not actionable if the small model must
    # discover every receiver and argument by blind paging. Resolve short,
    # exact-type producer evidence from the *already indexed* target bytecode.
    # The absence of a producer is only missing evidence, not proof that an
    # object cannot be built by composing a constructor or builder.
    from .java_operation_ir import parameter_types
    from .construction_state import symbol_record
    from .construction_api_admission import _words
    goal_words = set(_words(intent))
    input_types = []
    for candidate in result["target_api"][:3]:
        key = candidate.get("symbol_id")
        api = index.symbols.get(key, {})
        for type_ in parameter_types(api)[:5]:
            base = type_.split("[", 1)[0]
            if base in index.symbols and base not in input_types:
                input_types.append(base)
            if len(input_types) >= 6:
                break
        if len(input_types) >= 6:
            break
    for type_ in input_types:
        producers = []
        for key in sorted(
            index.member_result_keys.get(type_, ()),
            key=lambda name: (
                {"CONSTRUCTOR": 0, "METHOD": 1, "FIELD": 2}.get(
                    index.symbols.get(name, {}).get("kind"), 3),
                name,
            ),
        ):
            row = index.symbols.get(key, {})
            if row.get("kind") not in {"METHOD", "CONSTRUCTOR", "FIELD"}:
                continue
            if row.get("kind") != "CONSTRUCTOR" and not (
                row.get("static") or row.get("access", 0) & 8
            ):
                continue
            # A static EntityType.ARROW is type-correct but cannot create
            # the requested alien. Fields must be relevant to this feature,
            # not merely share the return type.
            if row.get("kind") == "FIELD" and not (
                set(_words(row.get("name", ""))) & goal_words
            ):
                continue
            if not index.accessible(key, constructor=row.get("kind") == "CONSTRUCTOR"):
                continue
            producers.append(symbol_record(key, row))
            if len(producers) >= 2:
                break
        result["typed_input_producers"].append({
            "required_type": type_, "verified_producers": producers,
        })
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
    target = context['target']
    if not isinstance(target, dict):
        raise ValueError('CONSTRUCTION_DIRECT_TARGET_INVALID')
    target_version = str(target.get('minecraft_version') or '').strip()
    target_loader = str(target.get('loader') or '').strip().casefold()
    if not target_version or target_loader != 'fabric':
        raise ValueError('CONSTRUCTION_DIRECT_TARGET_UNSUPPORTED')
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
            "feature_work_order": context.get("feature_work_order"),
            "feature_work_order_sha256": context.get("feature_work_order_sha256"),
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
                f"Implement exactly one independently owned Fabric {target_version} behavior "
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
            candidate = None
            try:
                candidate = (saved[behavior_id] if behavior_id in saved else
                             router.generate_java_source(
                                 "coder", messages,
                                 data_name="construction_direct_" + behavior_id[:60],
                                 source_frame=frame))
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
                from .construction_source_repair import preserve_failed_java, java_correction_message
                candidate = preserve_failed_java(error, candidate, kind="method", name=method,
                                                 owner=name, phase="direct_feature")
                if behavior_id in saved or attempt + 1 >= attempts:
                    raise
                messages.append(java_correction_message(error, candidate, frame, feature=True))
        if member is None:
            raise ValueError("CONSTRUCTION_DIRECT_BODY_NOT_VALIDATED")
        if behavior_id not in saved:
            # Draft state lives only inside this construction attempt. Do not
            # publish it as a reusable checkpoint until the COMPLETE owner
            # source has passed the exact target's Java compiler.
            saved[behavior_id] = member
        print("CONSTRUCTION_MEMBER_PROGRESS: " + json.dumps({
            "owner_symbol": owner + "#" + method + "()",
            "rendered_java_bytes": len(member.encode("utf-8")),
            "checkpointed": True, "target_compiled": False,
            "author_route": "direct_java",
        }, ensure_ascii=False), flush=True)
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
    from .compiled_feature_patch import replay_compiled_repair
    source = replay_compiled_repair(root, spec["target_path"], source)
    _parse(source)
    if not context.get("java_classpath"):
        raise ValueError("CONSTRUCTION_DIRECT_TARGET_CLASSPATH_REQUIRED")
    from .javac_bridge import analyze_java
    try:
        analyze_java(
            source, classpath=context["java_classpath"],
            java_version=str(context["target"]["java_version"]),
            filename=name + ".java",
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise ValueError(
            "CONSTRUCTION_DIRECT_FEATURE_TARGET_SYMBOL_INVALID: " + str(exc)[:2400]
        ) from exc
    if context.get("verify_source"):
        context["verify_source"](source, spec["target_path"])
    # An interrupted or rejected whole-feature candidate never becomes a
    # checkpoint, even when an individual method's immutable frame was valid.
    if saved and set(saved) - set(_read_checkpoint(checkpoint, identity)):
        _save_checkpoint(checkpoint, identity, saved)
    from .construction_coverage import record_authored_source_calls
    record_authored_source_calls(
        root, spec["target_path"], source, source_kind="direct")
    return source
