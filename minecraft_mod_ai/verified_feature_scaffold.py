"""Classpath-proven Fabric feature scaffold contract.

The host emits only lifecycle / declared interface syntax that has been
inspected in the SELECTED project's bytecode. Gameplay registration code stays
a model-authored *unverified* obligation until compiled and tested.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .feature_work_order import FeatureWorkOrder


def _ident(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", text))


def verified_fabric_feature_scaffold(
    work_order: FeatureWorkOrder,
    inspected_symbols: Mapping[str, Mapping[str, Any]],
    *,
    package_name: str, class_name: str,
) -> dict[str, Any]:
    if work_order.loader != "fabric":
        raise ValueError("FEATURE_SCAFFOLD_LOADER_UNSUPPORTED")
    if not all(_ident(part) for part in package_name.split(".")) or not _ident(class_name):
        raise ValueError("FEATURE_SCAFFOLD_JAVA_IDENTIFIER_INVALID")
    interface_name = "net.fabricmc.api.ModInitializer"
    iface = inspected_symbols.get(interface_name, {})
    if iface.get("kind") not in {"CLASS", "INTERFACE"}:
        raise ValueError("FEATURE_SCAFFOLD_FABRIC_INTERFACE_NOT_INSPECTED")
    signature = [
        (key, row) for key, row in inspected_symbols.items()
        if row.get("owner") == interface_name and row.get("kind") == "METHOD"
        and row.get("name") == "onInitialize"
        and row.get("return_type") == "void" and not row.get("parameters")
    ]
    if len(signature) != 1:
        raise ValueError("FEATURE_SCAFFOLD_LIFECYCLE_UNVERIFIED")
    symbol, method = signature[0]
    identity = str(iface.get("source_hash") or "")
    method_hash = str(method.get("source_hash") or "")
    if not identity or not method_hash:
        raise ValueError("FEATURE_SCAFFOLD_BYTECODE_IDENTITY_MISSING")
    path = "src/main/java/" + package_name.replace(".", "/") + "/" + class_name + ".java"
    if work_order.target_path != path:
        raise ValueError("FEATURE_SCAFFOLD_PATH_MISMATCH")
    # This method intentionally does not emit a fake successful registry
    # implementation. It is an insertion point that must be filled by the
    # feature author before build/acceptance can pass.
    source = (
        f"package {package_name};\n\n"
        f"public final class {class_name} implements {interface_name} {{\n"
        "    @Override\n"
        "    public void onInitialize() {\n"
        "    }\n"
        "}\n"
    )
    return {
        "schema_version": "mmm/verified-feature-scaffold-v1",
        "target_path": path,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "interface_symbol": interface_name,
        "method_symbol": symbol,
        "classpath_hashes": [identity, method_hash],
        "work_order_sha256": work_order.fingerprint(),
        "implementation_status": "SCAFFOLD_ONLY_UNIMPLEMENTED",
        "target_compile_passed": False,
        "runtime_verified": False,
    }
