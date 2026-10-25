"""File-level, compiler-feedback Java construction for the exact Fabric target.

This is the default construction author. Unlike the old operation graph, one
model response is an editable *compilation unit*, not a prospective IR edge.
Failures retain their source and compiler diagnostics for structural rewrites.
No code is checkpointed or certified before target symbol resolution succeeds.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .construction_direct_writer import (
    _claim_checkpoint,
    _digest,
    _read_checkpoint,
    _save_checkpoint,
)


class ConstructionFileError(ValueError):
    """A complete source unit could not be produced and verified."""


def _only_java(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ConstructionFileError("CONSTRUCTION_FILE_SOURCE_MISSING")
    source = text.strip()
    # Small chat-tuned models sometimes emit a fictitious file-write tool
    # invocation despite enable_tools=False. Extract exactly one embedded
    # compilation unit, then apply all normal identity/parser/compile gates.
    # A tool-only response is never treated as an executable action.
    if not source.startswith("package ") and source.count("<parameter=content>") == 1:
        embedded = source.split("<parameter=content>", 1)[1].split("</parameter>", 1)[0].strip()
        if embedded.startswith("package "):
            source = embedded
    if source.startswith("```"):
        lines = source.splitlines()
        if len(lines) < 3 or not lines[-1].strip() == "```":
            raise ConstructionFileError("CONSTRUCTION_FILE_FENCE_INVALID")
        source = "\n".join(lines[1:-1]).strip()
    if "```" in source:
        raise ConstructionFileError("CONSTRUCTION_FILE_EXTRA_CONTENT")
    if not source.startswith("package "):
        raise ConstructionFileError("CONSTRUCTION_FILE_NON_JAVA_RESPONSE")
    return source.rstrip() + "\n"


def _lexical_code(source: str) -> str:
    """Hide comments and strings when checking host-owned structural contracts.

    This is not a replacement for the Java parser; it prevents a model from
    satisfying a required initializer/method using a comment or string literal.
    """
    out = list(source)
    state = 'code'
    idx = 0
    while idx < len(source):
        two = source[idx:idx + 2]
        three = source[idx:idx + 3]
        if state == 'code':
            if two == '//':
                state = 'line'
                out[idx] = out[idx + 1] = ' '
                idx += 2
                continue
            if two == '/*':
                state = 'block'
                out[idx] = out[idx + 1] = ' '
                idx += 2
                continue
            if three == '"""':
                state = 'text_block'
                out[idx:idx + 3] = '   '
                idx += 3
                continue
            if source[idx] in ('"', "'"):
                state = source[idx]
                out[idx] = ' '
        elif state == 'line':
            if source[idx] == '\n':
                state = 'code'
            else:
                out[idx] = ' '
        elif state == 'block':
            if two == '*/':
                state = 'code'
                out[idx] = out[idx + 1] = ' '
                idx += 2
                continue
            if source[idx] != '\n':
                out[idx] = ' '
        elif state == 'text_block':
            if three == '"""':
                state = 'code'
                out[idx:idx + 3] = '   '
                idx += 3
                continue
            if source[idx] != '\n':
                out[idx] = ' '
        else:
            if source[idx] == '\\':
                out[idx] = ' '
                if idx + 1 < len(source):
                    out[idx + 1] = ' '
                idx += 2
                continue
            if source[idx] == state:
                state = 'code'
            out[idx] = ' '
        idx += 1
    return ''.join(out)


def _method_body(code: str, name: str) -> str | None:
    """Extract a zero-argument method body from structurally masked Java."""
    pattern = r"\b(?:void)\s+" + re.escape(name) + r"\s*\(\s*\)\s*\{"
    match = re.search(pattern, code)
    if not match:
        return None
    start = match.end()
    depth = 1
    for pos in range(start, len(code)):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
            if depth == 0:
                return code[start:pos]
    return None


def _check_identity(source: str, package: str, name: str, *, reject_placeholders=True) -> None:
    code = _lexical_code(source)
    package_decls = re.findall(r"(?m)^\s*package\s+([\w.]+)\s*;", code)
    if package_decls != [package]:
        raise ConstructionFileError("CONSTRUCTION_FILE_PACKAGE_MISMATCH")
    pattern = r"\bpublic\s+(?:(?:final|abstract)\s+)?class\s+" + re.escape(name) + r"\b"
    if not re.search(pattern, code):
        raise ConstructionFileError("CONSTRUCTION_FILE_PUBLIC_CLASS_MISMATCH")
    if reject_placeholders and re.search(r"\b(?:TODO|FIXME|UnsupportedOperationException)\b", source):
        raise ConstructionFileError("CONSTRUCTION_FILE_UNIMPLEMENTED_PLACEHOLDER")
    if not re.search(r"\{[^}]*[;{]", code, re.DOTALL):
        raise ConstructionFileError("CONSTRUCTION_FILE_EMPTY_CLASS")


def _check_concrete_entity(source: str, context: dict, name: str) -> None:
    """A generated ENTITY_EXISTS type must be instantiable, not a vacant shell."""
    contract = context.get("authoring_contract") or {}
    if contract.get("canonical_leaf") != "minecraft/entity/type":
        return
    code = _lexical_code(source)
    declaration = re.search(
        r"\bclass\s+" + re.escape(name) +
        r"\s+extends\s+([A-Za-z_$][\w$.]*)\b", code,
    )
    if declaration is None or declaration.group(1) in {"Object", "java.lang.Object"}:
        raise ConstructionFileError("CONSTRUCTION_ENTITY_SUPERCLASS_REQUIRED")
    if re.search(r'\bEntityType\s*\.\s*Builder\b', code):
        raise ConstructionFileError(
            'CONSTRUCTION_ENTITY_REGISTRY_OUTSIDE_TYPE: This file only implements the concrete '
            'entity subclass and its constructor. Remove EntityType.Builder/factory/registry '
            'creation here; the dependent minecraft/entity/registry artifact owns registration.')
    if not re.search(r"\b" + re.escape(name) + r"\s*\([^)]*\)\s*\{", code):
        raise ConstructionFileError("CONSTRUCTION_ENTITY_CONSTRUCTOR_REQUIRED")


def _check_entrypoint(source: str, spec: dict) -> None:
    """Check the actual lifecycle body, not arbitrary comments/string matches."""
    if not spec.get('bindings', {}).get('request_feature_entrypoint'):
        return
    code = _lexical_code(source)
    behaviors = json.loads(spec.get('bindings', {}).get('construction_behaviors_json') or '[]')
    method = 'onInitializeClient' if str(spec.get('side', '')).upper() == 'CLIENT' else 'onInitialize'
    interface = 'ClientModInitializer' if method.endswith('Client') else 'ModInitializer'
    if not re.search(r'\bimplements\b[^{}]*\b' + interface + r'\b', code):
        raise ConstructionFileError('CONSTRUCTION_FILE_ENTRYPOINT_INTERFACE_MISSING')
    lifecycle = _method_body(code, method)
    if lifecycle is None:
        raise ConstructionFileError('CONSTRUCTION_FILE_ENTRYPOINT_METHOD_MISSING')
    for behavior in behaviors:
        _check_behavior_registration(code, lifecycle, behavior)


def _check_behavior_registration(code: str, lifecycle: str, behavior: dict) -> None:
    """Enforce method, nonempty effect, lifecycle call, and coverage receipt."""
    key = str(behavior['behavior_id'])
    registrar = 'registerBehavior' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]
    member_body = _method_body(code, registrar)
    if member_body is None or not member_body.strip():
        raise ConstructionFileError('CONSTRUCTION_FILE_BEHAVIOR_METHOD_MISSING: ' + key)
    # An empty statement or an immediate bare return is not gameplay.
    # Compile-time validity alone cannot certify behavior, but it must not
    # admit the most obvious no-effect registrar implementation.
    if re.fullmatch(r'(?:\s*return\s*;|\s*;)*\s*', member_body):
        raise ConstructionFileError('CONSTRUCTION_FILE_BEHAVIOR_NOOP: ' + key)
    if not re.search(r'\bthis\s*\.\s*' + registrar + r'\s*\(\s*\)\s*;', lifecycle):
        raise ConstructionFileError('CONSTRUCTION_FILE_BEHAVIOR_REGISTRATION_MISSING: ' + key)
    # Downstream construction_coverage.record_constructed_requirement
    # requires these literal call/registration strings. Match that exact
    # receipt contract here, before persisting a misleading checkpoint.
    if registrar + '()' not in code or 'this.' + registrar + '();' not in lifecycle:
        raise ConstructionFileError('CONSTRUCTION_FILE_COVERAGE_SIGNATURE_MISMATCH: ' + key)



def _read_materialized_context(context: dict, *, target_path: str, source_context=()) -> list[dict]:
    root = Path(context["project_root"]).resolve()
    rows = []
    for item in source_context:
        if not isinstance(item, dict):
            continue
        relative = item.get("path")
        if not isinstance(relative, str) or not relative.endswith(".java"):
            continue
        absolute = (root / relative).resolve()
        if not absolute.is_relative_to(root) or not absolute.is_file() or absolute.is_symlink():
            continue
        if relative == target_path or absolute.stat().st_size > 120_000:
            continue
        rows.append({"path": relative, "authority": "materialized_prerequisite",
                     "code": absolute.read_text(encoding="utf-8")[:3500]})
        if len(rows) >= 4:
            return rows
    # Workspace paths are already materialized prerequisites, never donor code
    # fetched from an external mod repository.
    for relative, units in list(context.get("source_pages", {}).items())[:18]:
        if not isinstance(units, (list, tuple)):
            continue
        for item in units[:2]:
            if not isinstance(item, dict):
                continue
            code = item.get("code")
            if isinstance(code, str) and code.strip():
                rows.append({"path": str(relative), "signature": str(item.get("signature", "")),
                             "authority": str(item.get("authority", "source_only")),
                             "code": code[:3500]})
            if len(rows) >= 4:
                return rows
    # The full project will be verified by the normal production build after
    # emission; this bounded view is only a prompt context for the 9B model.
    return rows


def _file_api_intent(context: dict, intent: str, *, feature_entrypoint=False) -> str:
    if feature_entrypoint:
        # Feature entrypoints own gameplay behavior, not only registration glue.
        return intent
    contract = context.get('authoring_contract') or {}
    leaf = str(contract.get('canonical_leaf') or '')
    if leaf == 'minecraft/entity/type':
        # The concrete subclass must use real Entity/Mob/Monster constructors
        # from the selected 26.x classpath, not entity registration methods.
        return 'entity subclass constructor Entity Mob Monster EntityType Level'
    if leaf.startswith('minecraft/') and leaf.endswith('/registry'):
        # Registry is a host-owned operation, not the generic instruction
        # "Implement". Look up an inspected register callable on this domain.
        # The result is still admitted by exact classpath symbols, never by a
        # guessed API signature. Do not feed the unrelated game feature here.
        domain = leaf.split('/')[1]
        return ' '.join(('register', domain, 'in registry',
            *map(str, contract.get('host_requirements', {}).get('symbols', ()))))
    if leaf and contract.get('responsibility'):
        # Other narrow files retain their original host-owned responsibility.
        return ' '.join((str(contract['responsibility']), leaf,
            *map(str, contract.get('host_requirements', {}).get('symbols', ()))))
    return intent


def _project_file_evidence(evidence: dict, members, symbols: dict, intent: str) -> dict:
    from .construction_file_api_evidence import prompt_symbol_record
    owners = [symbols.get(row.get('symbol_id'), {}).get('owner', '')
              for row in evidence.get('target_api', ())]
    evidence['class_members'] = members.records(owners, intent=intent)
    evidence['target_api'] = [prompt_symbol_record(row) for row in evidence.get('target_api', ())]
    for inputs in evidence.get('typed_input_producers', ()):
        inputs['verified_producers'] = [prompt_symbol_record(row)
                                       for row in inputs.get('verified_producers', ())]
    return evidence


def _evidence(context: dict, intent: str, *, target_path: str, owner: str, router,
              feature_entrypoint=False) -> dict:
    from .construction_direct_writer import _evidence as _scoped_api_evidence
    from .construction_file_api_evidence import FileAPIEvidence
    from .construction_retrieval import ConstructionEvidenceIndex
    if (not feature_entrypoint and
            (context.get('authoring_contract') or {}).get('canonical_leaf') == 'minecraft/entity/type'):
        # Registration and concrete subclass construction are distinct graph
        # responsibilities. Do not ask the coder to choose a registry builder
        # while its task is only the entity class's usable super constructor.
        return FileAPIEvidence(context['symbols'], target_path=target_path,
                               owner=owner).entity_type_evidence()
    intent = _file_api_intent(context, intent, feature_entrypoint=feature_entrypoint)
    index = ConstructionEvidenceIndex(context["symbols"], context.get("source_pages", {}),
        router, target_path=target_path, owner=owner,
        authoring_contract=context.get("authoring_contract", {}))
    evidence = _scoped_api_evidence(index, intent, context.get("source_pages", {}))
    members = FileAPIEvidence(context['symbols'], target_path=target_path,
                              owner=owner, by_owner=index.by_owner)
    return _project_file_evidence(evidence, members, context['symbols'], intent)


def _diagnostic_api_evidence(context: dict, source: str, diagnostic: str) -> list[dict]:
    """Provide exact indexed candidate symbols after a compiler failure.

    Neither synthesize missing Minecraft APIs nor equate a lexical similarity
    with proof of behavior. The project symbol table is authoritative for the
    selected target classpath. The next model call receives concrete owners,
    fields and method signatures, not an unbounded search task.
    """
    if 'cannot find symbol' not in diagnostic and 'JAVA_PROJECT_ANALYSIS_FAILED' not in diagnostic:
        return []
    from .construction_file_api_evidence import FileAPIEvidence
    code = _lexical_code(source)
    package = re.search(r'\bpackage\s+([\w.]+)\s*;', code)
    name = re.search(r'\bclass\s+([\w$]+)', code)
    owner = (package.group(1) + '.' + name.group(1)) if package and name else ''
    index = FileAPIEvidence(context.get('symbols') or {},
        target_path=context.get('target_path', 'src/main/java/'), owner=owner)
    return index.records(index.source_owners(code, diagnostic), source=code, diagnostic=diagnostic)


def _verify_full_source(source: str, spec: dict, context: dict, *, package: str, name: str,
                        source_context=()) -> None:
    # A placeholder and a nonexistent Minecraft API are two different failures.
    # Check structural identity first, then compile to capture concrete missing
    # signatures. Only then reject semantic placeholders. Otherwise MiMo receives
    # the same generic placeholder error and reproduces identical broken Java.
    _check_identity(source, package, name, reject_placeholders=False)
    _check_concrete_entity(source, context, name)
    from .java_region_parser import _parse
    _parse(source)
    classpath = context.get("java_classpath")
    if not classpath:
        raise ConstructionFileError("CONSTRUCTION_FILE_TARGET_CLASSPATH_REQUIRED")
    from .construction_project_compile import verify_java_source_with_dependencies
    registered_ui = []
    if spec.get("bindings", {}).get("request_feature_entrypoint") and context.get("symbols"):
        from .source_owned_ui_event import read_verified_typed_ui_events
        registered_ui = read_verified_typed_ui_events(
            project_root=context["project_root"], symbols=context["symbols"],
            behaviors=json.loads(spec.get("bindings", {}).get("construction_behaviors_json") or "[]"),
            package=package, side=str(spec.get("side") or "COMMON").upper())
    # The event is in a separate generated namespace. Include every concrete
    # server producer source in this compile, including its startup transport;
    # discovery of same-package siblings cannot find these dependencies.
    source_context = [*source_context, *(
        {"path": path} for event in registered_ui for path in event["producer_source_paths"]
    )]
    # Compile the full owner and its materialized sibling sources, rather than
    # a detached method or an owner without its actual project prerequisites.
    compiled_sources = verify_java_source_with_dependencies(source, context=context,
        target_path=spec["target_path"], filename=name + ".java",
        source_context=source_context)
    _check_identity(source, package, name)
    _check_entrypoint(source, spec)
    if spec.get("bindings", {}).get("request_feature_entrypoint") and context.get("symbols"):
        # An initializer that calls an invented registrar is NOT a gameplay
        # hook. The exact source and all project prerequisites have already
        # compiled. Now require an actual installed Fabric callback field
        # from the same verified target classpath (also on checkpoint replay).
        from .host_fabric_callback_abi import callback_interfaces
        from .feature_registration_proof import validate_registered_event_sources
        checked_events = callback_interfaces(
            context["symbols"], side=str(spec.get("side") or "COMMON"))
        checked_behaviors = json.loads(
            spec.get("bindings", {}).get("construction_behaviors_json") or "[]")
        from .source_owned_custom_event import read_verified_custom_event_receipts
        registered_custom = read_verified_custom_event_receipts(
            project_root=context["project_root"], symbols=context["symbols"],
            behavior_ids=[row["behavior_id"] for row in checked_behaviors],
            package=package, side=str(spec.get("side") or "COMMON").upper(),
        )
        registered_custom.extend(registered_ui)
        validate_registered_event_sources(
            source, checked_behaviors, checked_events,
            materialized_events=registered_custom)
    verify_source = context.get("verify_source")
    if verify_source:
        verify_source(source, spec["target_path"])
    from .java_source_snapshot import record_compiled_sources
    record_compiled_sources(context, source=source,
        target_path=spec["target_path"], sources=compiled_sources)


def _required_entrypoint_bindings(spec: dict, behavior_specs) -> dict:
    client_side = str(spec.get("side", "")).upper() == "CLIENT"
    return {
        "required_registrations": [
            {"behavior_id": str(row["behavior_id"]),
             "method_name": "registerBehavior" + hashlib.sha256(
                 str(row["behavior_id"]).encode("utf-8")).hexdigest()[:20],
             "intent": row["intent"]}
            for row in behavior_specs
        ],
        "required_entrypoint": {
            "interface": ("net.fabricmc.api.ClientModInitializer" if client_side
                          else "net.fabricmc.api.ModInitializer"),
            "method": "onInitializeClient" if client_side else "onInitialize",
        },
    }


def compose_complete_source(router, spec: dict, context: dict, *, package: str, name: str,
                            owner: str, behaviors=(), source_context=()) -> str:
    """Create a complete Java class and fix actual source, not inferred node IDs."""
    if str(context["target"].get("loader", "")).casefold() != "fabric":
        raise ConstructionFileError("CONSTRUCTION_FILE_FABRIC_REQUIRED")
    if not context.get("java_classpath") or not context.get("project_root"):
        raise ConstructionFileError("CONSTRUCTION_FILE_WORKSPACE_REQUIRED")
    root = Path(context["project_root"]).resolve()
    if not root.is_dir():
        raise ConstructionFileError("CONSTRUCTION_FILE_WORKSPACE_NOT_FOUND")
    identifier = _digest({
        "spec": spec, "owner": owner, "target": context["target"],
        "prerequisites": context.get("prerequisite_hashes", {}),
        "work_order": context.get("feature_work_order_sha256", ""),
        "source_author_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    checkpoint = root / ".mmm" / "construction_files" / (identifier + ".json")
    with _claim_checkpoint(checkpoint):
        saved = _read_checkpoint(checkpoint, identifier)
        checkpoint_error = None
        if saved:
            if set(saved) != {"source"} or not isinstance(saved["source"], str):
                raise ConstructionFileError("CONSTRUCTION_FILE_CHECKPOINT_INVALID")
            try:
                _verify_full_source(saved["source"], spec, context, package=package, name=name,
                                    source_context=source_context)
            except (ValueError, RuntimeError, OSError) as exc:
                # A previously compilable owner may become invalid when a
                # prerequisite source changes. Repair, never blindly trust it
                # and never lose the previous complete source/diagnostic.
                checkpoint_error = str(exc)
            else:
                return saved["source"]
        requirement = str(spec.get("requirement", ""))
        behavior_specs = list(behaviors or ())
        contract = context.get("authoring_contract") or {}
        leaf_responsibility = str(contract.get("responsibility") or "").strip()
        # The user feature goal is not the assignment of a registry/constructor
        # leaf. A generated registry must implement registration, not "fight
        # aliens" or whole-game features. Keep the goal for traceability.
        scoped_requirement = (
            leaf_responsibility if contract.get("canonical_leaf") and leaf_responsibility
            and not spec.get("bindings", {}).get("request_feature_entrypoint")
            else requirement
        )
        prompt = {
            "task": "Implement the complete Minecraft/Fabric Java compilation unit",
            "requirement": scoped_requirement,
            "original_feature_goal": requirement,
            "assigned_behaviors": behavior_specs,
            "feature_work_order": context.get("feature_work_order"),
            "target": context["target"],
            "package": package, "public_class": name,
            "output_path": spec["target_path"],
            "side": spec.get("side", "COMMON"),
            "construction_requirements": context.get("authoring_contract", {}),
            "construction_bindings": context.get("binding_pages", {}),
            "assigned_runtime_records": spec.get("bindings", {}).get("owned_runtime_records_json", "[]"),
            "retained_effect_ownership": spec.get("bindings", {}).get("retained_typed_behavior_json", "{}"),
            "implementation_requirements": spec.get("bindings", {}).get("java_implementation_requirements", ""),
            "inspected_target_api": _evidence(context, requirement + " " +
                " ".join(str(row.get("intent", "")) for row in behavior_specs),
                target_path=spec["target_path"], owner=owner, router=router,
                feature_entrypoint=bool(spec.get("bindings", {}).get("request_feature_entrypoint"))),
            "materialized_project_source": _read_materialized_context(
                context, target_path=spec["target_path"], source_context=source_context),
        }
        if spec.get("bindings", {}).get("request_feature_entrypoint"):
            prompt.update(_required_entrypoint_bindings(spec, behavior_specs))
        system = (
            "You implement real Java source files for the EXACT selected Fabric and Minecraft versions. "
            "Return ONLY the complete Java file, starting with its package statement. "
            "You have NO shell, file-write, tool-call or code-execution tools. Never emit "
            "<tool_call>, <function=...>, prose, XML, or a filename as your answer. "
            "The response itself must begin with 'package ' and contain Java source only. "
            "The selected mappings use Minecraft OFFICIAL names, not Yarn names; "
            "do not import net.minecraft.entity or net.minecraft.registry from Yarn. "
            "Use exactly the specified package and public class; include all necessary imports, "
            "members, constructors, registry fields, lifecycle and required behavior. "
            "Implement functionality directly, without another planning response, node graph, stub, "
            "TODO, fake API, silent success, or external mod source reuse. "
            "A listed API is evidence only when its exact owner/signature is verified. "
            "class_members contains inspected overloads, nested builders and factory contracts; "
            "use those exact parameter types rather than recalling an API from another version. "
            "Never invent a Minecraft registry or method. The assigned canonical_leaf is the "
            "only Java responsibility of this file; the original_feature_goal is context, "
            "not permission to implement every game feature in the registry. "
            "If materialized_project_source lacks a required user-defined entity type, "
            "do not reference an imaginary class, fabricate a factory overload, or "
            "substitute an UnsupportedOperationException implementation. "
            "If required_registrations exist, implement each named zero-argument registration method "
            "and invoke this.method_name() from onInitialize/onInitializeClient. The project verifies "
            "their exact identities. Implement their effects rather than empty placeholders. "
            "For each repair, rewrite the WHOLE compilable file, including corrections to bad design "
            "and imports, not just the line of the compiler error."
        )
        from .construction_file_prompt import file_messages, generate_file_text
        repair = None
        if checkpoint_error is not None:
            repair = {
                "repair_complete_file": True,
                "error": checkpoint_error,
                "previous_source": saved["source"],
                "instruction": "The previously accepted compilation unit no longer resolves. Repair the WHOLE source against the new project context."
            }
        messages = file_messages(system, prompt, repair)
        previous = {hashlib.sha256(saved['source'].encode()).hexdigest()} if checkpoint_error is not None else set()
        last_error = "No candidate received"
        # This is an execution safety boundary, not the remedy for a bad
        # implementation. Actual distinct compiler errors trigger source
        # rewrites; repeat sources are rejected immediately. Keep compiler
        # repair attempts consistent with the immutable feature work order.
        acceptance = (context.get('feature_work_order') or {}).get('acceptance') or {}
        repair_budget = acceptance.get('max_compiler_repair_attempts', 2)
        if type(repair_budget) is not int or repair_budget < 0:
            raise ConstructionFileError('CONSTRUCTION_FILE_REPAIR_CONTRACT_INVALID')
        for attempt in range(1 + repair_budget):
            raw = generate_file_text(router, messages, target_path=spec['target_path'])
            # Carry any compacted discovery view forward instead of restoring
            # the oversized initial observations after a compiler rejection.
            prompt = json.loads(messages[1]['content'])
            source = None
            try:
                source = _only_java(raw)
                fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()
                if fingerprint in previous:
                    raise ConstructionFileError("CONSTRUCTION_FILE_NO_SOURCE_PROGRESS")
                previous.add(fingerprint)
                # Source corrections must reach the verifier before checkpointing.
                _verify_full_source(source, spec, context, package=package, name=name,
                    source_context=source_context)
            except (ValueError, RuntimeError, OSError) as exc:
                last_error = str(exc)
                if last_error.startswith("CONSTRUCTION_FILE_NO_SOURCE_PROGRESS"):
                    raise ConstructionFileError(last_error) from exc
                # Full source, not just an exception summary, is provided to the
                # model to support structural repair. Save failed-source trace
                # locally for reconstruction; never checkpoint it as success.
                trace = root / ".mmm" / "construction_file_failures"
                trace.mkdir(parents=True, exist_ok=True)
                failed = trace / (identifier + "-" + str(attempt) + ".json")
                failed.write_text(json.dumps({"path": spec["target_path"], "error": last_error,
                    "source": source if source is not None else raw}, ensure_ascii=False, indent=2))
                repair = {
                    "repair_complete_file": True, "error": last_error,
                    "previous_source": source if source is not None else raw,
                    "required_behavior": requirement,
                    "verified_symbols_from_compiler_failure": _diagnostic_api_evidence(
                        context, source or '', last_error),
                    "instruction": "Fix the cause, including architecture/dependency/actual API mistakes. Only the listed inspected symbols are verified; missing prerequisite source may require structural changes. Return the entire Java file."
                }
                messages = file_messages(system, prompt, repair)
                continue
            from .construction_coverage import record_authored_source_calls
            record_authored_source_calls(root, spec["target_path"], source, source_kind="direct")
            _save_checkpoint(checkpoint, identifier, {"source": source})
            print("CONSTRUCTION FILE AUTHOR: " + json.dumps({
                "target_path": spec["target_path"], "owner": owner, "attempts": attempt + 1,
                "target_symbols_verified": True, "runtime_verified": False,
                "source_sha256": fingerprint,
            }, ensure_ascii=False), flush=True)
            return source
        raise ConstructionFileError("CONSTRUCTION_FILE_UNRESOLVED: " + last_error[:1200])
