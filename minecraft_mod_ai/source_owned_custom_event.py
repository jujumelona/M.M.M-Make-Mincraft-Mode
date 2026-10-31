"""Materialize a real Fabric custom event from an explicitly inspected source site.

A Java method's presence is evidence of a possible injection site, NOT proof
that it represents the requested gameplay occurrence. The caller must supply
an explicitly bound site symbol, and runtime acceptance must still exercise
the user's original trigger/effects. No default Minecraft event is substituted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from collections.abc import Mapping

from .java_operation_ir import parameter_types


_CLASS = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+$")
_PRIMITIVE = {
    "void": "V", "boolean": "Z", "byte": "B", "char": "C",
    "short": "S", "int": "I", "long": "J", "float": "F", "double": "D",
}


def _descriptor(type_name: str) -> str:
    kind = str(type_name)
    if kind.endswith("[]"):
        return "[" + _descriptor(kind[:-2])
    if kind in _PRIMITIVE:
        return _PRIMITIVE[kind]
    if not _CLASS.fullmatch(kind):
        raise ValueError("SOURCE_SITE_JVM_TYPE_INVALID: " + kind)
    return "L" + kind.replace(".", "/") + ";"


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".mmm-source-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _verified_method(symbols: Mapping, site_ref: str) -> dict:
    row = symbols.get(site_ref)
    if not isinstance(row, Mapping) or row.get("kind") != "METHOD":
        raise ValueError("SOURCE_SITE_METHOD_NOT_INSPECTED: " + site_ref)
    owner, name = row.get("owner"), row.get("name")
    if (not isinstance(owner, str) or not _CLASS.fullmatch(owner)
            or not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z_$][\w$]*", name) is None
            or name in {"<init>", "<clinit>"}):
        raise ValueError("SOURCE_SITE_OWNER_OR_METHOD_INVALID")
    access = int(row.get("access") or 0)
    # No static method, native/abstract bytecode or synthetic bridge can be
    # injected by the receiver-bound host adapter.
    if (access & (8 | 256 | 1024 | 4096)) or row.get("static"):
        raise ValueError("SOURCE_SITE_NOT_INJECTABLE")
    parent = symbols.get(owner)
    if not isinstance(parent, Mapping) or parent.get("kind") != "CLASS":
        # Bytecode symbol tables may key classes by an opaque symbol ID.
        # Require their actual inspected CLASS entry, not a string guess.
        parent = next((value for value in symbols.values()
                       if isinstance(value, Mapping)
                       and value.get("kind") == "CLASS"
                       and value.get("owner") == owner), None)
    if parent is None:
        raise ValueError("SOURCE_SITE_OWNER_NOT_INSPECTED")
    args = list(parameter_types(row))
    ret = str(row.get("return_type") or "")
    descriptor = "(" + "".join(_descriptor(value) for value in args) + ")" + _descriptor(ret)
    checked = dict(row)
    checked.update(owner=owner, name=name, arguments=args, result=ret, descriptor=descriptor)
    return checked


def plan_verified_custom_event(
    *, symbols: Mapping, source_site_ref: str, behavior_id: str,
    package: str, mod_id: str, at: str = "HEAD", side: str = "COMMON",
    java_version: int = 25,
) -> dict:
    """Return exact source files and an ABI; this function does not write."""
    if not behavior_id or not isinstance(behavior_id, str):
        raise ValueError("SOURCE_SITE_BEHAVIOR_REQUIRED")
    if not _CLASS.fullmatch(package + ".Generated"):
        raise ValueError("SOURCE_SITE_PACKAGE_INVALID")
    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,63}", mod_id):
        raise ValueError("SOURCE_SITE_MOD_ID_INVALID")
    if at not in {"HEAD", "RETURN"}:
        raise ValueError("SOURCE_SITE_INJECTION_POINT_INVALID")
    if side not in {"COMMON", "CLIENT"}:
        raise ValueError("SOURCE_SITE_SIDE_INVALID")
    if type(java_version) is not int or java_version not in {17, 21, 25}:
        raise ValueError("SOURCE_SITE_JAVA_COMPATIBILITY_INVALID")
    site = _verified_method(symbols, source_site_ref)
    if (str(site.get("side") or "COMMON").upper() == "CLIENT"
            and side != "CLIENT"):
        raise ValueError(
            "SOURCE_SITE_SIDE_MISMATCH: client-only method cannot be "
            "injected from a common server Mixin"
        )
    digest = hashlib.sha256(
        (behavior_id + "\0" + source_site_ref + "\0" + at).encode()
    ).hexdigest()[:16]
    event_name, mixin_name = "MmmEvent" + digest, "MmmMixin" + digest
    event_owner = package + "." + event_name
    mixin_file = mod_id + ".mmm_" + digest + ".mixins.json"
    args = [site["owner"], *site["arguments"]]
    declarations = ", ".join(
        type_name + " v" + str(i) for i, type_name in enumerate(args)
    )
    names = ", ".join("v" + str(i) for i in range(len(args)))
    call_args = ", ".join(
        type_name + " arg" + str(i) for i, type_name in enumerate(site["arguments"])
    )
    invoke_args = ", ".join(
        ["(" + site["owner"] + ")(Object)this"]
        + ["arg" + str(i) for i in range(len(site["arguments"]))]
    )
    callback_kind = ("org.spongepowered.asm.mixin.injection.callback.CallbackInfo"
                     if site["result"] == "void" else
                     "org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable<?>")
    callback_signature = ", ".join(filter(None, [call_args, callback_kind + " callback"]))
    source = f"""package {package};

// MMM:INSPECTED_SOURCE_EVENT:{digest}
public final class {event_name} {{
    private {event_name}() {{}}
    @FunctionalInterface public interface Listener {{
        void onEvent({declarations});
    }}
    public static final net.fabricmc.fabric.api.event.Event<Listener> EVENT =
        net.fabricmc.fabric.api.event.EventFactory.createArrayBacked(
            Listener.class, listeners -> ({names}) -> {{
                for (Listener listener : listeners) listener.onEvent({names});
            }}
        );
}}
"""
    injection = f"""package {package}.mixin;

@org.spongepowered.asm.mixin.Mixin({site['owner']}.class)
public abstract class {mixin_name} {{
    @org.spongepowered.asm.mixin.injection.Inject(
        method = "{site['name']}{site['descriptor']}",
        at = @org.spongepowered.asm.mixin.injection.At("{at}"),
        require = 1
    )
    private void mmmObserved({callback_signature}) {{
        {event_owner}.EVENT.invoker().onEvent({invoke_args});
    }}
}}
"""
    config = {
        "required": True, "package": package + ".mixin",
        "compatibilityLevel": "JAVA_" + str(java_version),
        ("client" if side == "CLIENT" else "mixins"): [mixin_name],
        "injectors": {"defaultRequire": 1},
    }
    package_path = package.replace(".", "/")
    source_root = "src/client/java" if side == "CLIENT" else "src/main/java"
    files = {
        f"{source_root}/{package_path}/{event_name}.java": source,
        f"{source_root}/{package_path}/mixin/{mixin_name}.java": injection,
        f"src/main/resources/{mixin_file}":
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    }
    return {
        "schema": "mmm/inspected-custom-source-v1",
        "behavior_id": behavior_id,
        "source_site_ref": source_site_ref,
        "source_descriptor": site["descriptor"],
        "mixin_config": mixin_file,
        "files": files,
        "callback": {
            "behavior_id": behavior_id,
            "event_id": event_owner + "#EVENT", "event_owner": event_owner,
            "event_field": "EVENT", "event_interface": event_owner + ".Listener",
            "method": "onEvent", "parameters": args, "return_type": "void",
            "type_parameters": [], "throws": [],
            "provenance": "verified_source_site_mixin",
        },
        "runtime_verified": False,
    }


from contextlib import contextmanager

_PRODUCER_LOCK_LOCAL = threading.local()


@contextmanager
def _producer_cross_process_lock(path: Path):
    """Serialize manifest/receipt publication on POSIX and Windows.

    The project-level thread gate is process-local; a second Python process
    can still update the same fabric.mod.json. Keep the on-disk lock in both
    platforms and release it even on validation or filesystem failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = str(path.resolve())
    held = getattr(_PRODUCER_LOCK_LOCAL, "held", None)
    if held is None:
        held = {}
        _PRODUCER_LOCK_LOCAL.held = held
    if identity in held:
        held[identity] += 1
        try:
            yield
        finally:
            held[identity] -= 1
        return
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        held[identity] = 1
        try:
            yield
        finally:
            held.pop(identity, None)
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def _checked_source_target(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("SOURCE_SITE_PATH_UNSAFE")
    # resolve() erases symlink provenance. An in-project symlink with matching
    # bytes is not an independently materialized producer source file.
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError("SOURCE_SITE_PATH_UNSAFE")
    return path


def _previous_custom_receipts(folder, current_digest):
    """Exclude the active receipt; the publisher checks its exact bytes."""
    return sorted(path for path in folder.glob("*.json")
                  if path.name != current_digest + ".json")


def _obsolete_custom_sources(root, symbols, metadata, behavior_id, package,
                             current_digest):
    """Validate and identify older owned Mixins for the SAME behavior.

    Replacing a source-site binding must also retire its receipt, injectors
    and fabric.mod.json entry. Removing an unverified or edited file is never
    permitted; such a conflict needs explicit manual reconciliation.
    """
    folder = _checked_source_target(root, ".mmm/source-producers")
    if not folder.is_dir():
        return []
    mixins = metadata.get("mixins", [])
    result = []
    for path in _previous_custom_receipts(folder, current_digest):
        if path.is_symlink():
            raise ValueError("SOURCE_SITE_RECEIPT_SYMLINK")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != (
                "mmm/inspected-custom-source-receipt-v1"):
            raise ValueError("SOURCE_SITE_RECEIPT_SCHEMA_INVALID")
        if data.get("behavior_id") != behavior_id or data.get("package") != package:
            continue
        old_digest = hashlib.sha256((
            behavior_id + "\0" + data["source_site_ref"] + "\0" + data["at"]
        ).encode()).hexdigest()[:16]
        if path.stem != old_digest:
            raise ValueError("SOURCE_SITE_RECEIPT_ID_MISMATCH")
        if old_digest == current_digest:
            continue
        old = plan_verified_custom_event(
            symbols=symbols, source_site_ref=data["source_site_ref"],
            behavior_id=behavior_id, package=package, mod_id=metadata["id"],
            at=data["at"], side=data["side"], java_version=data["java_version"],
        )
        if old["mixin_config"] not in {
            entry if isinstance(entry, str) else entry.get("config")
            for entry in mixins
        }:
            raise ValueError("SOURCE_SITE_OLD_MIXIN_NOT_REGISTERED")
        for name, contents in old["files"].items():
            target = _checked_source_target(root, name)
            if not target.is_file() or target.read_text(encoding="utf-8") != contents:
                raise ValueError("SOURCE_SITE_OBSOLETE_SOURCE_CONFLICT: " + name)
        result.append((path, path.read_text(encoding="utf-8"), old))
    return result


def materialize_verified_custom_event(
    *, project_root: str | Path, symbols: Mapping, source_site_ref: str,
    behavior_id: str, package: str, at: str = "HEAD", side: str = "COMMON",
    java_version: int = 25,
) -> dict:
    """Install event, Mixin and manifest atomically per process, idempotently.

    Call only for an explicitly assigned, inspected source method. Runtime
    injection verification and behavioral playtesting remain mandatory.
    """
    root = Path(project_root).resolve()
    manifest = root / "src/main/resources/fabric.mod.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_REQUIRED")
    from .project_write_lock import project_write_lock
    lock = root / ".mmm/source-producer.lock"
    with project_write_lock(root), _producer_cross_process_lock(lock):
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
            raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_INVALID")
        mixins = metadata.get("mixins", [])
        if not isinstance(mixins, list) or any(not (
                isinstance(x, str) or
                (isinstance(x, dict) and isinstance(x.get("config"), str)
                 and x.get("environment") in {"client", "server", "*"})
        ) for x in mixins):
            raise ValueError("SOURCE_SITE_FABRIC_MIXINS_INVALID")
        result = plan_verified_custom_event(
            symbols=symbols, source_site_ref=source_site_ref,
            behavior_id=behavior_id, package=package, mod_id=metadata["id"],
            at=at, side=side, java_version=java_version,
        )
        # Validate the COMPLETE publication contract before writing the first
        # source file. A pre-existing incompatible receipt must never leave a
        # new Mixin in fabric.mod.json without matching provenance.
        digest = hashlib.sha256(
            (behavior_id + "\0" + source_site_ref + "\0" + at).encode()
        ).hexdigest()[:16]
        receipt = {
            "schema": "mmm/inspected-custom-source-receipt-v1",
            "behavior_id": behavior_id, "source_site_ref": source_site_ref,
            "package": package, "side": side, "java_version": java_version,
            "at": at,
        }
        receipt_path = root / ".mmm/source-producers" / (digest + ".json")
        expected_receipt = json.dumps(receipt, sort_keys=True, ensure_ascii=False) + "\n"
        obsolete = _obsolete_custom_sources(
            root, symbols, metadata, behavior_id, package, digest)
        if receipt_path.is_symlink():
            raise ValueError("SOURCE_SITE_RECEIPT_SYMLINK")
        if (receipt_path.exists() and
                receipt_path.read_text(encoding="utf-8") != expected_receipt):
            raise ValueError("SOURCE_SITE_RECEIPT_CONFLICT")
        for relative, contents in result["files"].items():
            path = _checked_source_target(root, relative)
            if path.exists() and path.read_text(encoding="utf-8") != contents:
                raise ValueError("SOURCE_SITE_EXISTING_FILE_CONFLICT: " + relative)
        original_manifest = manifest.read_text(encoding="utf-8")
        manifest_entry = ({"config": result["mixin_config"], "environment": "client"}
                          if side == "CLIENT" else result["mixin_config"])
        old_config_names = {old["mixin_config"] for _, _, old in obsolete}
        active = [entry for entry in mixins if
                  (entry if isinstance(entry, str) else entry.get("config"))
                  not in old_config_names]
        matching = [entry for entry in active
                    if (entry if isinstance(entry, str) else entry.get("config"))
                    == result["mixin_config"]]
        if matching and any(entry != manifest_entry for entry in matching):
            raise ValueError("SOURCE_SITE_MIXIN_ENVIRONMENT_MISMATCH")
        if len(matching) > 1:
            raise ValueError("SOURCE_SITE_MIXIN_REGISTRATION_DUPLICATE")
        if not matching:
            active.append(manifest_entry)
        manifest_changed = active != mixins
        if manifest_changed:
            metadata["mixins"] = active
        new_manifest = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
                        if manifest_changed else original_manifest)
        created_files = []
        receipt_created = False
        try:
            for relative, contents in result["files"].items():
                path = root / relative
                if not path.exists():
                    # Register rollback ownership before the atomic replace:
                    # a mocked/failed writer can raise just after os.replace.
                    created_files.append((path, contents))
                    _atomic_write(path, contents)
            if manifest_changed:
                _atomic_write(manifest, new_manifest)
            if not receipt_path.exists():
                receipt_created = True
                _atomic_write(receipt_path, expected_receipt)
            for old_receipt, _, old_plan in obsolete:
                for relative, contents in old_plan["files"].items():
                    old_target = _checked_source_target(root, relative)
                    if old_target.is_file() and old_target.read_text(encoding="utf-8") == contents:
                        old_target.unlink()
                    elif old_target.exists():
                        raise ValueError("SOURCE_SITE_OBSOLETE_SOURCE_CHANGED: " + relative)
                old_receipt.unlink()
        except Exception:
            # One failed write cannot publish a partially registered Mixin.
            # The two locks exclude concurrent publishers while rolling back.
            # Restore only exact bytes we just wrote; do not erase a
            # conflicting update performed outside this transaction.
            incomplete = []
            if receipt_created and receipt_path.is_file():
                if receipt_path.read_text(encoding="utf-8") == expected_receipt:
                    receipt_path.unlink()
                else:
                    incomplete.append(str(receipt_path))
            # Restore deleted previous producers before removing the new one.
            for old_receipt, old_receipt_bytes, old_plan in obsolete:
                for relative, contents in old_plan["files"].items():
                    old_target = _checked_source_target(root, relative)
                    if not old_target.exists():
                        _atomic_write(old_target, contents)
                    elif old_target.read_text(encoding="utf-8") != contents:
                        incomplete.append(str(old_target))
                if not old_receipt.exists():
                    _atomic_write(old_receipt, old_receipt_bytes)
                elif old_receipt.read_text(encoding="utf-8") != old_receipt_bytes:
                    incomplete.append(str(old_receipt))
            if manifest_changed and manifest.is_file():
                current = manifest.read_text(encoding="utf-8")
                if current == new_manifest:
                    _atomic_write(manifest, original_manifest)
                elif current != original_manifest:
                    incomplete.append(str(manifest))
            for path, contents in reversed(created_files):
                if path.is_file() and path.read_text(encoding="utf-8") == contents:
                    path.unlink()
                elif path.exists():
                    incomplete.append(str(path))
            if incomplete:
                raise RuntimeError(
                    "SOURCE_SITE_PUBLICATION_ROLLBACK_INCOMPLETE: "
                    + ", ".join(incomplete))
            raise
    return result



def read_verified_custom_event_receipts(
    *, project_root: str | Path, symbols: Mapping, behavior_ids,
    package: str, side: str = "COMMON",
    include_producer_paths: bool = False,
) -> list[dict]:
    """Reproduce each generated producer from its inspected origin and bytes.

    A receipt by itself is never valid source evidence. Stale or tampered Java,
    missing Mixin configuration, uninspected sites and mismatched mod metadata
    are rejected before an event may satisfy registration proof.
    """
    root = Path(project_root).resolve()
    manifest = root / "src/main/resources/fabric.mod.json"
    if not manifest.is_file():
        return []
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("id"), str):
        raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_INVALID")
    present = config.get("mixins", [])
    if not isinstance(present, list):
        raise ValueError("SOURCE_SITE_FABRIC_MIXINS_INVALID")
    names = {row if isinstance(row, str) else row.get("config")
             for row in present if isinstance(row, (dict, str))}
    allowed = set(behavior_ids)
    folder = _checked_source_target(root, ".mmm/source-producers")
    if not folder.is_dir():
        return []
    checked = {}
    for receipt_path in folder.glob("*.json"):
        if receipt_path.is_symlink():
            raise ValueError("SOURCE_SITE_RECEIPT_SYMLINK")
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != "mmm/inspected-custom-source-receipt-v1":
            raise ValueError("SOURCE_SITE_RECEIPT_SCHEMA_INVALID")
        behavior_id = data.get("behavior_id")
        if behavior_id not in allowed or data.get("package") != package or data.get("side") != side:
            continue
        if behavior_id in checked:
            raise ValueError("SOURCE_SITE_RECEIPT_DUPLICATE_BEHAVIOR")
        fingerprint = hashlib.sha256((
            behavior_id + chr(0) + data["source_site_ref"] + chr(0) + data["at"]
        ).encode()).hexdigest()[:16]
        if receipt_path.name != fingerprint + ".json":
            raise ValueError("SOURCE_SITE_RECEIPT_ID_MISMATCH")
        expected = plan_verified_custom_event(
            symbols=symbols, source_site_ref=data["source_site_ref"],
            behavior_id=behavior_id, package=package,
            mod_id=config["id"], at=data["at"], side=side,
            java_version=data["java_version"],
        )
        if expected["mixin_config"] not in names:
            raise ValueError("SOURCE_SITE_MIXIN_NOT_REGISTERED")
        # Fabric can list a config on the wrong environment while its source
        # and receipt bytes remain untouched. In that case the intended
        # client/server event NEVER fires even though Java compilation passes.
        registration = [
            entry for entry in present
            if (entry if isinstance(entry, str) else entry.get("config"))
            == expected["mixin_config"]
        ]
        if side == "CLIENT":
            admitted = any(isinstance(entry, dict)
                           and entry.get("environment") == "client"
                           for entry in registration)
        else:
            admitted = any(
                isinstance(entry, str)
                or (isinstance(entry, dict)
                    and entry.get("environment") in {"server", "*"})
                for entry in registration)
        if not admitted:
            raise ValueError("SOURCE_SITE_MIXIN_ENVIRONMENT_MISMATCH")
        for name, contents in expected["files"].items():
            path = _checked_source_target(root, name)
            if not path.is_file():
                raise ValueError("SOURCE_SITE_MATERIALIZED_FILE_MISSING")
            if path.read_text(encoding="utf-8") != contents:
                raise ValueError("SOURCE_SITE_MATERIALIZED_FILE_MISMATCH")
        callback = dict(expected["callback"])
        if include_producer_paths:
            callback["producer_source_paths"] = sorted(
                path for path in expected["files"] if path.endswith(".java"))
        checked[behavior_id] = callback
    return list(checked.values())
