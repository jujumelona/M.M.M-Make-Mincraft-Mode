"""Target-scoped, actual Minecraft GameTest evidence for registry-only content.

This is not a Mineflayer or client-visual substitute. Only the item/block and
shaped-recipe content profile may use this server GameTest route; any dynamic
gameplay, client, state, multiplayer or scripted effect requires external
runtime scenario receipts.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_START = "// MMM_MANAGED_CONTENT_GAMETEST_V1 START"
_END = "// MMM_MANAGED_CONTENT_GAMETEST_V1 END"
_ALLOWED_MODULE_KINDS = frozenset({"item", "block", "recipe", "typed_host"})


def _safe_file(path: Path) -> Path | None:
    if path.is_symlink():
        return None
    try:
        candidate = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return candidate if candidate.is_file() else None


def _eligible_modules(approved: Any) -> tuple[tuple[str, str], ...]:
    """Return only content whose live behavior we can actually assert."""
    modules = getattr(approved, "modules", ())
    lock = getattr(getattr(approved, "base_proposal", None), "spec", None)
    platform = getattr(lock, "platform", None)
    if (
        platform is None
        or str(getattr(platform, "loader", "")).casefold() != "fabric"
        or str(getattr(platform, "mappings_kind", "")).casefold()
           not in {"mojang", "official", "official_mojang"}
        or not re.fullmatch(r"2[6-9]\.\d+(?:\.\d+)?", str(getattr(platform, "minecraft_version", "")))
    ):
        return ()
    entries = [
        (str(getattr(module, "module_id", "")),
         str(getattr(module, "kind", "")).casefold())
        for module in modules
    ]
    if not entries or any(kind not in _ALLOWED_MODULE_KINDS for _, kind in entries):
        return ()
    content = sorted((name, kind) for name, kind in entries if kind in {"item", "block"})
    if not content or any(not _IDENTIFIER.fullmatch(name) for name, _ in content):
        return ()
    # A typed host program is not necessarily passive. Do not certify its
    # arbitrary behavior; only allow this route for a saved, content-only
    # authored build where the remaining implemented modules are registry,
    # block or recipe artifacts.
    if any(kind == "typed_host" and name != "authored_typed_plan" for name, kind in entries):
        return ()
    return tuple(content)


def _manifest(approved: Any, root: Path) -> dict[str, Any] | None:
    content = _eligible_modules(approved)
    if not content:
        return None
    spec = approved.base_proposal.spec
    mod_id = str(spec.mod_id)
    pkg = str(spec.package_name)
    if not _IDENTIFIER.fullmatch(mod_id) or not pkg:
        return None
    resources = root / "src/main/resources"
    for name, kind in content:
        if kind == "item":
            location = resources / "assets" / mod_id / "items" / f"{name}.json"
        else:
            location = resources / "assets" / mod_id / "blockstates" / f"{name}.json"
        location = _safe_file(location)
        if location is None:
            return None
        try:
            if not isinstance(json.loads(location.read_text(encoding="utf-8")), dict):
                return None
        except (UnicodeError, json.JSONDecodeError, OSError):
            return None
    # A recipe is a static artifact and must pass the separate resource
    # validator; a registry GameTest cannot attest to crafting behavior.
    # Recipes are *not* allowed on this limited runtime proof route.
    kinds = {str(getattr(m, "kind", "")).casefold() for m in approved.modules}
    if "recipe" in kinds:
        return None
    return {"mod_id": mod_id, "package": pkg, "content": content}


def _java_assertions(manifest: Mapping[str, Any]) -> str:
    mod_id = str(manifest["mod_id"])
    lines = [f"        {_START}"]
    for name, kind in manifest["content"]:
        registry = "ITEM" if kind == "item" else "BLOCK"
        name = str(name)
        lines.extend((
            "        if (!net.minecraft.core.registries.BuiltInRegistries."
            + registry + ".containsKey("
            + "net.minecraft.resources.ResourceLocation.fromNamespaceAndPath("
            + f'"{mod_id}", "{name}"))) {{',
            f'            throw new AssertionError("GameTest missing live {kind} registry entry: {mod_id}:{name}");',
            "        }",
        ))
    lines.append(f"        {_END}")
    return "\n".join(lines)


def eligible_for_managed_runtime(approved: Any) -> bool:
    """May plan a content GameTest; full eligibility checks follow generation."""
    kinds = {str(getattr(m, "kind", "")).casefold() for m in getattr(approved, "modules", ())}
    return bool(_eligible_modules(approved)) and "recipe" not in kinds


def install_managed_content_gametest(root: Path, approved: Any) -> dict[str, Any] | None:
    """Inject testable in-server behavior BEFORE validation and clean Gradle build."""
    manifest = _manifest(approved, root)
    if manifest is None:
        return None
    package = str(manifest["package"])
    main_class = "".join(part.capitalize() for part in manifest["mod_id"].split("_")) + "Mod"
    test_class = main_class + "GameTests"
    source = root / "src/gametest/java" / Path(*package.split(".")) / f"{test_class}.java"
    path = _safe_file(source)
    if path is None:
        return None
    text = path.read_text(encoding="utf-8")
    marker = "        context.succeed();"
    if marker not in text:
        return None
    if _START in text or _END in text:
        text = re.sub(
            r"(?m)^        // MMM_MANAGED_CONTENT_GAMETEST_V1 START\n"
            r".*?^        // MMM_MANAGED_CONTENT_GAMETEST_V1 END\n",
            "", text, flags=re.DOTALL,
        )
    block = _java_assertions(manifest)
    changed = text.replace(marker, block + "\n" + marker, 1)
    if changed == text or changed.count(_START) != 1 or changed.count(_END) != 1:
        return None
    path.write_text(changed, encoding="utf-8")
    return {
        "schema_version": "mmm/managed-content-gametest-v1",
        "status": "INSTALLED",
        "source": str(path.relative_to(root)),
        "content": [list(item) for item in manifest["content"]],
        "source_sha256": "sha256:" + hashlib.sha256(changed.encode()).hexdigest(),
    }


def independently_verified_managed_runtime(
    root: Path,
    approved: Any,
    build_report: Mapping[str, Any] | None,
    *,
    gametest_passed: bool,
) -> dict[str, Any] | None:
    """Fail closed unless the exact generated assertions ran in Minecraft.

    A passing arbitrary GameTest or a manually supplied PASS receipt is not
    sufficient; verify the same source, testcase, test execution, and report.
    """
    manifest = _manifest(approved, root)
    if manifest is None or not gametest_passed or not isinstance(build_report, Mapping):
        return None
    package = str(manifest["package"])
    main_class = "".join(part.capitalize() for part in manifest["mod_id"].split("_")) + "Mod"
    klass = main_class + "GameTests"
    source = root / "src/gametest/java" / Path(*package.split(".")) / f"{klass}.java"
    path = _safe_file(source)
    report_file = build_report.get("gametest_report")
    report = _safe_file(Path(report_file)) if isinstance(report_file, str) and report_file else None
    if path is None or report is None:
        return None
    text = path.read_text(encoding="utf-8")
    expected = _java_assertions(manifest)
    if text.count(expected) != 1 or text.count(_START) != 1 or text.count(_END) != 1:
        return None
    try:
        tree = ET.parse(report)
        cases = tuple(tree.getroot().iter("testcase"))
    except (ET.ParseError, OSError):
        return None
    expected_class = klass.casefold()
    expected_method = "generatedRegistriesAreLive".casefold()
    has_test = any(
        (
            str(case.attrib.get("name", "")).casefold()
            in {expected_method, f"{expected_class}.{expected_method}"}
            and str(case.attrib.get("classname", "")).casefold().split(".")[-1]
            in {"", expected_class}
        )
        or (
            "".join(ch for ch in str(case.attrib.get("name", "")).casefold() if ch.isalnum())
            .startswith("".join(ch for ch in manifest["mod_id"] if ch.isalnum()))
            and "".join(ch for ch in str(case.attrib.get("name", "")).casefold() if ch.isalnum())
            .endswith("generatedregistriesarelive")
        )
        for case in cases
    )
    if not has_test or any(
        case.find("failure") is not None or case.find("error") is not None
        or case.find("skipped") is not None
        for case in cases
    ):
        return None
    return {
        "schema_version": "mmm/managed-content-gametest-runtime-v1",
        "status": "PASS",
        "runtime_kind": "live_minecraft_server_gametest",
        "mod_id": manifest["mod_id"],
        "content": [list(item) for item in manifest["content"]],
        "source_sha256": "sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        "report_sha256": "sha256:" + hashlib.sha256(report.read_bytes()).hexdigest(),
    }
