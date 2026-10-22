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
    package: str, mod_id: str, at: str = "HEAD",
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
    site = _verified_method(symbols, source_site_ref)
    digest = hashlib.sha256(
        (behavior_id + "\0" + source_site_ref + "\0" + at).encode()
    ).hexdigest()[:16]
    event_name, mixin_name = "MmmEvent" + digest, "MmmMixin" + digest
    event_owner = package + "." + event_name
    mixin_owner = package + ".mixin." + mixin_name
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
        "compatibilityLevel": "JAVA_21",
        "mixins": [mixin_name], "injectors": {"defaultRequire": 1},
    }
    package_path = package.replace(".", "/")
    files = {
        f"src/main/java/{package_path}/{event_name}.java": source,
        f"src/main/java/{package_path}/mixin/{mixin_name}.java": injection,
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
            "event_id": event_owner + "#EVENT", "event_owner": event_owner,
            "event_field": "EVENT", "event_interface": event_owner + ".Listener",
            "method": "onEvent", "parameters": args, "return_type": "void",
            "type_parameters": [], "throws": [],
            "provenance": "verified_source_site_mixin",
        },
        "runtime_verified": False,
    }


def materialize_verified_custom_event(
    *, project_root: str | Path, symbols: Mapping, source_site_ref: str,
    behavior_id: str, package: str, at: str = "HEAD",
) -> dict:
    """Install event, Mixin and manifest atomically per process, idempotently.

    Call only for an explicitly assigned, inspected source method. Runtime
    injection verification and behavioral playtesting remain mandatory.
    """
    root = Path(project_root).resolve()
    manifest = root / "src/main/resources/fabric.mod.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_REQUIRED")
    import fcntl
    lock = root / ".mmm/source-producer.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
            raise ValueError("SOURCE_SITE_FABRIC_MANIFEST_INVALID")
        mixins = metadata.get("mixins", [])
        if not isinstance(mixins, list) or not all(isinstance(x, str) for x in mixins):
            raise ValueError("SOURCE_SITE_FABRIC_MIXINS_INVALID")
        result = plan_verified_custom_event(
            symbols=symbols, source_site_ref=source_site_ref,
            behavior_id=behavior_id, package=package, mod_id=metadata["id"], at=at,
        )
        for relative, contents in result["files"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or path.is_symlink():
                raise ValueError("SOURCE_SITE_PATH_UNSAFE")
            if path.exists() and path.read_text(encoding="utf-8") != contents:
                raise ValueError("SOURCE_SITE_EXISTING_FILE_CONFLICT: " + relative)
        for relative, contents in result["files"].items():
            path = root / relative
            if not path.exists():
                _atomic_write(path, contents)
        if result["mixin_config"] not in mixins:
            metadata["mixins"] = [*mixins, result["mixin_config"]]
            _atomic_write(
                manifest, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return result
