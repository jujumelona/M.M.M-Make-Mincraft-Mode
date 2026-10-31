"""Build a requested player action before its gameplay listener needs a hook.

The host owns the form, input codec, server actor, real event and startup
registration. Only the bounded input ABI is authored. Passive occurrences may
not be converted into player commands. These receipts prove wiring, not play.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from .source_owned_hook import SourceHookUnavailable

_ACTIVE = re.compile(
    r"\b(?:selects?|chooses?|clicks?|confirms?|initiates?|requests?|submits?|"
    r"press(?:es)?|interacts?|activates?)\b|선택|클릭|확인|요청|누르",
    re.IGNORECASE,
)
_ACTOR = "net.minecraft.commands.CommandSourceStack"
_SCHEMA = "mmm/typed-ui-occurrence-v1"
_PLAYER = re.compile(r"\b(?:player|user)\b|플레이어|사용자", re.IGNORECASE)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _player_activation(text):
    # Admit an action whose grammatical subject remains the player. Merely
    # finding "select" later in a world/server/negated clause would invent
    # manual activation for an automatic occurrence.
    for actor in _PLAYER.finditer(text):
        if re.search(r"\b(?:no|not|without)(?:\s+\w+){0,2}\s*$", text[:actor.start()], re.IGNORECASE):
            continue
        tail = text[actor.end():actor.end() + 256]
        if actor.group() in {"플레이어", "사용자"}:
            if not re.search(r"자동|서버|시스템|없이|않", tail) and _ACTIVE.search(tail):
                return True
            continue
        prefix = re.match(
            r"\s*(?:(?:manually|explicitly|actively|deliberately|then)\s+)*"
            r"(?:(?:must|can|may|will|should)\s+|needs?\s+to\s+)?", tail, re.IGNORECASE)
        actions = [_ACTIVE.match(tail, prefix.end())]
        for conjunction in re.finditer(
                r"\band\s+(?:(?:then|manually|explicitly|actively|deliberately)\s+)*", tail, re.IGNORECASE):
            before = tail[:conjunction.start()]
            if not re.search(r"\b(?:without|not|never|while|unless|until|server|system)\b", before, re.IGNORECASE):
                actions.append(_ACTIVE.match(tail, conjunction.end()))
        for action in (item for item in actions if item is not None):
            rest = re.split(r"[.;!?]|\band\b", tail[action.end():], maxsplit=1, flags=re.IGNORECASE)[0]
            if not re.search(r"\b(?:automatically|autonomously|randomly|without)\b", rest, re.IGNORECASE):
                return True
    return False


def _require_target(behavior, symbols, minecraft_version):
    if (str(minecraft_version) != "26.2"
            or any(not any(isinstance(row, Mapping) and row.get("kind") == "CLASS"
                           and row.get("owner") == owner for row in symbols.values())
                   for owner in (_ACTOR, "net.fabricmc.fabric.api.event.Event",
                                 "net.fabricmc.fabric.api.event.EventFactory",
                                 "net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback"))):
        raise SourceHookUnavailable(
            str(behavior.get("behavior_id") or ""),
            reason="original_occurrence_requires_non_ui_source_owner", inspected=0)


def _author_explicit_activation(router, behavior):
    """Retain the established explicit-control decision and recipe ABI."""
    from .fixed_template_generation import _validate_native_arguments

    schema = {"type": "object", "properties": {
        "activation": {"type": "string", "enum": ["typed_ui_action", "unavailable"]},
        "trigger_quote": {"type": "string", "maxLength": 512},
    }, "required": ["activation", "trigger_quote"], "additionalProperties": False}
    response = router.generate_structured_data(
        "planner", [{"role": "system", "content": (
            "Determine whether the ORIGINAL trigger explicitly requires the player "
            "to select, confirm or initiate this action. Only then choose typed_ui_action "
            "and quote that exact active part of trigger. The host will implement that "
            "interaction's form and server transport. Arrival, ticking, damage, automatic "
            "progress and world conditions alone are NOT player UI actions. Never invent "
            "a command/item/key trigger to replace them; choose unavailable instead. "
            "All preconditions and gameplay effects remain the original contract.")},
            {"role": "user", "content": _canonical(dict(behavior))}],
        data_name="source_owned_ui_activation", parameters=schema,
    )
    chosen = _validate_native_arguments(response, schema)
    quote = chosen["trigger_quote"].strip()
    if (chosen["activation"] != "typed_ui_action" or not quote
            or quote not in str(behavior.get("trigger") or "") or not _player_activation(quote)):
        raise SourceHookUnavailable(
            str(behavior.get("behavior_id") or ""),
            reason="original_trigger_has_no_owned_ui_activation", inspected=0)


def _activation_evidence(router, behavior):
    if _player_activation(str(behavior.get("trigger") or "")):
        _author_explicit_activation(router, behavior)
        return None
    # An eligibility condition is not necessarily the requested action.
    # Interpret the full immutable behavior; never use the legacy vocabulary
    # as evidence that all other player actions are unsupported.
    from .player_action_evidence import author_player_action_evidence

    return author_player_action_evidence(router, behavior)


def _action_record(behavior, evidence):
    record = {"trigger": behavior["trigger"], "server_action": str(behavior["intent"]),
              "original_behavior": dict(behavior)}
    if evidence is not None:
        record["activation_evidence"] = evidence
    return record


def _require_activation(behavior, activation_evidence):
    if activation_evidence is None:
        # Preserve the validation and byte identity of existing v1 recipes.
        if not _player_activation(str(behavior.get("trigger") or "")):
            raise SourceHookUnavailable(
                str(behavior.get("behavior_id") or ""),
                reason="original_occurrence_requires_non_ui_source_owner", inspected=0)
    else:
        from .player_action_evidence import validate_player_action_evidence

        validate_player_action_evidence(behavior, activation_evidence)


def _occurrence_recipe(behavior, metadata, package, mod_id, minecraft_version, activation_evidence):
    recipe = {"schema": _SCHEMA, "behavior": deepcopy(dict(behavior)),
              "action_metadata": metadata, "package": package,
              "mod_id": mod_id, "minecraft_version": str(minecraft_version)}
    if activation_evidence is not None:
        recipe["activation_evidence"] = deepcopy(dict(activation_evidence))
    return recipe


def author_typed_ui_occurrence(router, behavior, *, symbols, package, mod_id,
                               minecraft_version):
    """Plan an original player action and preserve its complete conditions."""
    _require_target(behavior, symbols, minecraft_version)
    from .typed_action_authoring import author_action_metadata

    evidence = _activation_evidence(router, behavior)
    metadata = author_action_metadata(router, _action_record(behavior, evidence),
                                     requested_prompt=str(behavior.get("source_quote") or ""))
    return plan_typed_ui_occurrence(
        behavior=behavior, action_metadata=metadata, activation_evidence=evidence,
        package=package, mod_id=mod_id, minecraft_version=minecraft_version, symbols=symbols,
    )


def plan_typed_ui_occurrence(*, behavior, action_metadata, package, mod_id,
                             minecraft_version, symbols, activation_evidence=None):
    """Pure renderer for a concrete form -> validated server input -> event ABI."""
    _require_target(behavior, symbols, minecraft_version)
    _require_activation(behavior, activation_evidence)
    from .typed_action_authoring import validate_action_metadata
    from .typed_action_client import render_action_client_java
    from .typed_action_java import (
        render_action_command_registration,
        render_action_dispatch_java,
    )

    if (not isinstance(behavior.get("behavior_id"), str) or not behavior["behavior_id"]
            or not re.fullmatch(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", package)
            or not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", mod_id)):
        raise ValueError("UI_OCCURRENCE_IDENTITY_INVALID")
    metadata = validate_action_metadata(action_metadata)
    recipe = _occurrence_recipe(behavior, metadata, package, mod_id, minecraft_version, activation_evidence)
    digest = hashlib.sha256(_canonical(recipe).encode()).hexdigest()[:20]
    namespace = package + ".occurrence" + digest
    action = {"id": "occurrence_" + digest, "label": metadata["intent"],
              "inputs": metadata["inputs"]}
    function = {"id": "apply_occurrence", "return_type": "int", "parameters": [
        {"type": "object", "name": "source"},
        *({"type": row["type"], "name": row["name"]} for row in action["inputs"]),
    ]}
    binding = {"event": "command", "function": function["id"], "config": {
        "literal": "mmm_occurrence_" + digest, "permission_level": 0, "action": action}}
    plan = {"functions": [function], "event_bindings": [binding]}
    types = [_ACTOR, *("java.lang.String" if row["type"] == "string" else row["type"]
                       for row in action["inputs"])]
    from .java_operation_ir import KEYWORDS
    names = ["source"]
    for index, row in enumerate(action["inputs"]):
        name = row["name"]
        if name in KEYWORDS:
            name = "actionInput" + str(index)
            while name in names or any(other["name"] == name for other in action["inputs"]):
                name += "_"
        names.append(name)
    declarations = ", ".join(typ + " " + name for typ, name in zip(types, names))
    arguments = ", ".join(names)
    # The player identity is supplied by the server command context. No input
    # field names a state key, receiver, function or permission source.
    values = [f'({_ACTOR}) values.get("source")']
    for row in action["inputs"]:
        cast = {"string": "String", "boolean": "Boolean", "int": "Integer",
                "long": "Long", "double": "Double"}[row["type"]]
        values.append(f'({cast}) values.get({json.dumps(row["name"])})')
    event = f'''package {namespace};

public final class Occurrence {{
    private Occurrence() {{}}
    @FunctionalInterface public interface Listener {{ boolean onAction({declarations}); }}
    public static final net.fabricmc.fabric.api.event.Event<Listener> EVENT =
        net.fabricmc.fabric.api.event.EventFactory.createArrayBacked(
            Listener.class, listeners -> ({arguments}) -> {{
                for (Listener listener : listeners) {{
                    if (listener.onAction({arguments})) return true;
                }}
                return false;
            }});
}}
'''
    program = f'''package {namespace};

public final class AuthoredProgram {{
    private AuthoredProgram() {{}}
    public static Object invoke(String function, java.util.Map<String,Object> values) {{
        if (!"apply_occurrence".equals(function)) throw new IllegalArgumentException("unknown function");
        return Occurrence.EVENT.invoker().onAction({", ".join(values)}) ? 1 : 0;
    }}
}}
'''
    registration = "\n".join(render_action_command_registration(
        binding, minecraft_version=str(minecraft_version)))
    initializer = f'''package {namespace};

public final class OccurrenceInitializer implements net.fabricmc.api.ModInitializer {{
    @Override public void onInitialize() {{
{registration}
    }}
}}
'''
    common_path = "src/main/java/" + namespace.replace(".", "/") + "/"
    files = {common_path + "Occurrence.java": event,
             common_path + "AuthoredProgram.java": program,
             common_path + "AuthoredActions.java": render_action_dispatch_java(plan, package=namespace),
             common_path + "OccurrenceInitializer.java": initializer,
             "src/client/java/" + namespace.replace(".", "/") + "/AuthoredActionsClient.java":
                 render_action_client_java(plan, package=namespace,
                     mod_id=mod_id + "_" + digest, minecraft_version=str(minecraft_version))}
    return {"schema": _SCHEMA, "behavior_id": behavior["behavior_id"], "recipe": recipe,
            "digest": digest, "action": action, "files": files,
            "entrypoints": {"main": namespace + ".OccurrenceInitializer",
                            "client": namespace + ".AuthoredActionsClient"},
            "callback": {"event_id": namespace + ".Occurrence#EVENT",
                "behavior_id": behavior["behavior_id"],
                "event_owner": namespace + ".Occurrence", "event_field": "EVENT",
                "event_interface": namespace + ".Occurrence.Listener", "method": "onAction",
                "parameters": types, "parameter_names": names, "return_type": "boolean",
                "type_parameters": [], "throws": [], "provenance": "host_typed_ui_action",
                "producer_source_paths": sorted(path for path in files if path.startswith("src/main/java/")),
                "action": action},
            "runtime_verified": False}


def _path(root, relative):
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError("UI_OCCURRENCE_PATH_UNSAFE")
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("UI_OCCURRENCE_PATH_SYMLINK")
    return candidate


def _manifest(root):
    path = _path(root, "src/main/resources/fabric.mod.json")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
        raise ValueError("UI_OCCURRENCE_MANIFEST_INVALID")  # noqa: TRY004 - external contract
    entries = metadata.setdefault("entrypoints", {})
    if not isinstance(entries, dict):
        raise ValueError("UI_OCCURRENCE_ENTRYPOINTS_INVALID")  # noqa: TRY004 - external contract
    for side in ("main", "client"):
        if not isinstance(entries.setdefault(side, []), list):
            raise ValueError("UI_OCCURRENCE_ENTRYPOINTS_INVALID")  # noqa: TRY004 - external contract
    return path, metadata


def _render_recipe(recipe, symbols):
    return plan_typed_ui_occurrence(
        behavior=recipe["behavior"], action_metadata=recipe["action_metadata"],
        symbols=symbols, package=recipe["package"], mod_id=recipe["mod_id"],
        minecraft_version=recipe["minecraft_version"],
        activation_evidence=recipe.get("activation_evidence"))


def _obsolete_producers(root, planned, symbols):
    """Replace only unchanged host-owned output for this exact behavior owner."""
    folder = _path(root, ".mmm/typed-ui-producers")
    if not folder.is_dir():
        return []
    obsolete = []
    for receipt in sorted(folder.glob("*.json")):
        recipe = json.loads(_path(root, str(receipt.relative_to(root))).read_text(encoding="utf-8"))
        if recipe.get("schema") != _SCHEMA:
            raise ValueError("UI_OCCURRENCE_RECEIPT_SCHEMA_INVALID")
        if (recipe["package"] != planned["recipe"]["package"]
                or recipe["behavior"]["behavior_id"] != planned["behavior_id"]):
            continue
        previous = _render_recipe(recipe, symbols)
        if receipt.stem != previous["digest"]:
            raise ValueError("UI_OCCURRENCE_RECEIPT_ID_MISMATCH")
        if previous["digest"] == planned["digest"]:
            continue
        for relative, contents in previous["files"].items():
            path = _path(root, relative)
            if not path.is_file() or path.read_text(encoding="utf-8") != contents:
                raise ValueError("UI_OCCURRENCE_PREVIOUS_SOURCE_CONFLICT: " + relative)
        obsolete.append((receipt, previous))
    return obsolete


def materialize_typed_ui_occurrence(planned, *, project_root, symbols):
    """Install only host-rendered files and record the exact ownership recipe."""
    from .project_write_lock import project_write_lock
    from .source_owned_custom_event import _atomic_write, _producer_cross_process_lock

    expected = _render_recipe(planned["recipe"], symbols)
    if _canonical(planned) != _canonical(expected):
        raise ValueError("UI_OCCURRENCE_PLAN_MISMATCH")
    root = Path(project_root).resolve()
    # Mixin producers update the same manifest. One lock must protect both
    # transports, otherwise a parallel stale read can erase the other's entry.
    lock = _path(root, ".mmm/source-producer.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with project_write_lock(root), _producer_cross_process_lock(lock):
        manifest, metadata = _manifest(root)
        if metadata["id"] != planned["recipe"]["mod_id"]:
            raise ValueError("UI_OCCURRENCE_MOD_ID_MISMATCH")
        obsolete = _obsolete_producers(root, planned, symbols)
        receipt = _path(root, ".mmm/typed-ui-producers/" + planned["digest"] + ".json")
        expected_receipt = _canonical(planned["recipe"]) + "\n"
        if receipt.is_file() and receipt.read_text(encoding="utf-8") != expected_receipt:
            raise ValueError("UI_OCCURRENCE_RECEIPT_CONFLICT")
        for relative, contents in planned["files"].items():
            path = _path(root, relative)
            if path.exists() and path.read_text(encoding="utf-8") != contents:
                raise ValueError("UI_OCCURRENCE_EXISTING_FILE_CONFLICT: " + relative)

        before_manifest = manifest.read_text(encoding="utf-8")
        removed_files = []
        for old_receipt, previous in obsolete:
            for relative, contents in previous["files"].items():
                removed_files.append((_path(root, relative), contents))
            removed_files.append((old_receipt, old_receipt.read_text(encoding="utf-8")))

        for side, owner in planned["entrypoints"].items():
            entries = metadata["entrypoints"][side]
            superseded = {previous["entrypoints"][side] for _, previous in obsolete}
            entries[:] = [entry for entry in entries if (
                entry if isinstance(entry, str) else entry.get("value")
                if isinstance(entry, dict) else None) not in superseded]
            existing = {entry if isinstance(entry, str) else entry.get("value")
                        for entry in entries if isinstance(entry, (str, dict))}
            if owner not in existing:
                entries.append(owner)
        new_manifest = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"

        created_files = []
        receipt_created = not receipt.exists()
        try:
            for relative, contents in planned["files"].items():
                path = _path(root, relative)
                if not path.exists():
                    created_files.append((path, contents))
                    _atomic_write(path, contents)
            _atomic_write(manifest, new_manifest)
            for old_path, old_contents in removed_files:
                # Obsolete output was byte-checked above, before publication.
                if old_path.exists() and old_path.read_text(encoding="utf-8") == old_contents:
                    old_path.unlink()
                elif old_path.exists():
                    raise ValueError("UI_OCCURRENCE_OBSOLETE_CHANGED: " + str(old_path))
            if receipt_created:
                _atomic_write(receipt, expected_receipt)
        except Exception:
            # Restore exactly this transaction, without touching independent
            # producers or pre-existing entries. Both project and process
            # locks remain held throughout rollback.
            incomplete = []
            if receipt_created and receipt.is_file():
                if receipt.read_text(encoding="utf-8") == expected_receipt:
                    receipt.unlink()
                else:
                    incomplete.append(str(receipt))
            for old_path, old_contents in removed_files:
                if not old_path.exists():
                    _atomic_write(old_path, old_contents)
                elif old_path.read_text(encoding="utf-8") != old_contents:
                    incomplete.append(str(old_path))
            if manifest.is_file():
                current = manifest.read_text(encoding="utf-8")
                if current == new_manifest:
                    _atomic_write(manifest, before_manifest)
                elif current != before_manifest:
                    incomplete.append(str(manifest))
            for path, contents in reversed(created_files):
                if path.is_file() and path.read_text(encoding="utf-8") == contents:
                    path.unlink()
                elif path.exists():
                    incomplete.append(str(path))
            if incomplete:
                raise RuntimeError(
                    "UI_OCCURRENCE_PUBLICATION_ROLLBACK_INCOMPLETE: "
                    + ", ".join(incomplete))
            raise
    return planned


def read_verified_typed_ui_events(*, project_root, symbols, behaviors, package, side="COMMON"):
    """Re-render original contracts and check bytes plus both startup owners."""
    if side != "COMMON":
        return []
    root = Path(project_root).resolve()
    folder = _path(root, ".mmm/typed-ui-producers")
    if not folder.is_dir():
        return []
    _, metadata = _manifest(root)
    allowed = {row["behavior_id"]: row for row in behaviors}
    checked = {}
    for receipt in sorted(folder.glob("*.json")):
        recipe = json.loads(_path(root, str(receipt.relative_to(root))).read_text(encoding="utf-8"))
        if recipe.get("schema") != _SCHEMA:
            raise ValueError("UI_OCCURRENCE_RECEIPT_SCHEMA_INVALID")
        behavior = recipe["behavior"]
        identity = behavior["behavior_id"]
        if identity not in allowed or recipe["package"] != package:
            continue
        if _canonical(behavior) != _canonical(allowed[identity]):
            raise ValueError("UI_OCCURRENCE_BEHAVIOR_MISMATCH")
        if identity in checked:
            raise ValueError("UI_OCCURRENCE_RECEIPT_DUPLICATE_BEHAVIOR")
        if metadata["id"] != recipe["mod_id"]:
            raise ValueError("UI_OCCURRENCE_MOD_ID_MISMATCH")
        expected = _render_recipe(recipe, symbols)
        if receipt.stem != expected["digest"]:
            raise ValueError("UI_OCCURRENCE_RECEIPT_ID_MISMATCH")
        for side_name, owner in expected["entrypoints"].items():
            entries = metadata["entrypoints"][side_name]
            count = sum(
                (entry if isinstance(entry, str) else entry.get("value")) == owner
                for entry in entries if isinstance(entry, (str, dict)))
            if count == 0:
                raise ValueError("UI_OCCURRENCE_STARTUP_NOT_REGISTERED")
            if count != 1:
                raise ValueError("UI_OCCURRENCE_STARTUP_DUPLICATE")
        for relative, contents in expected["files"].items():
            path = _path(root, relative)
            if not path.is_file() or path.read_text(encoding="utf-8") != contents:
                raise ValueError("UI_OCCURRENCE_MATERIALIZED_FILE_MISMATCH")
        checked[identity] = expected["callback"]
    return list(checked.values())
