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
    recipes = sorted(
        str(getattr(module, "module_id", ""))
        for module in approved.modules
        if str(getattr(module, "kind", "")).casefold() == "recipe"
    )
    for recipe_id in recipes:
        if not _IDENTIFIER.fullmatch(recipe_id):
            return None
        path = _safe_file(resources / "data" / mod_id / "recipe" / f"{recipe_id}.json")
        if path is None:
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or "result" not in value:
                return None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
    # The final generated Java is more authoritative than a broad host facts
    # symbol table: mixed-mapping catalogs can describe an alias that the
    # actual compiled mod does not use. Lock verification to actual source.
    registry_owners: dict[str, str] = {}
    for kind, source_name in (("item", "ModItems.java"), ("block", "ModBlocks.java")):
        if kind not in {entry_kind for _, entry_kind in content}:
            continue
        path = _safe_file(
            root / "src/main/java" / Path(*pkg.split(".")) / "registry" / source_name
        )
        if path is None:
            return None
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
        owners = set(re.findall(
            r"net\.minecraft(?:\.[A-Za-z_][A-Za-z_0-9]*)+\.BuiltInRegistries",
            source,
        ))
        if len(owners) != 1:
            return None
        registry_owners[kind] = owners.pop()
    return {"mod_id": mod_id, "package": pkg, "content": content, "recipes": recipes,
            "registries": registry_owners}


def _java_assertions(manifest: Mapping[str, Any]) -> str:
    """Emit executable Minecraft-side runtime checks, using locked host owners.

    The approved host facts (not guessed Mojang/Yarn import paths) identify
    the registry owner. Java reflection is limited to the public runtime
    registry interface and fails the GameTest on any incompatible target.
    """
    mod_id = str(manifest["mod_id"])
    lines = [f"        {_START}", "        try {"]
    for name, kind in manifest["content"]:
        registry = "ITEM" if kind == "item" else "BLOCK"
        owner = str(manifest["registries"][kind])
        lines.extend((
            "            {",
            "                Object registry = Class.forName("
            + f'"{owner}").getField("{registry}").get(null);',
            "                if (!(registry instanceof Iterable<?>)) {",
            f'                    throw new AssertionError("Runtime {registry} registry is not iterable");',
            "                }",
            "                boolean present = false;",
            "                for (Object entry : (Iterable<?>) registry) {",
            "                    for (java.lang.reflect.Method getter : registry.getClass().getMethods()) {",
            "                        if (getter.getParameterCount() != 1",
            '                            || !(getter.getName().equals("getKey") || getter.getName().equals("getId"))) continue;',
            "                        try {",
            "                            Object key = getter.invoke(registry, entry);",
            f'                            if ("{mod_id}:{name}".equals(String.valueOf(key))) present = true;',
            "                        } catch (ReflectiveOperationException | IllegalArgumentException ignored) {",
            "                            // This overload cannot address registered entries.",
            "                        }",
            "                        if (present) break;",
            "                    }",
            "                    if (present) break;",
            "                }",
            f'                if (!present) throw new AssertionError("GameTest missing live {kind}: {mod_id}:{name}");',
            "            }",
        ))
    if manifest["recipes"]:
        lines.extend((
            "            {",
            '                Object level = context.getClass().getMethod("getLevel").invoke(context);',
            '                Object server = level.getClass().getMethod("getServer").invoke(level);',
            "                Object recipeAccess;",
            "                try {",
            '                    recipeAccess = level.getClass().getMethod("recipeAccess").invoke(level);',
            "                } catch (NoSuchMethodException missing) {",
            '                    recipeAccess = server.getClass().getMethod("getRecipeManager").invoke(server);',
            "                }",
            '                Class<?> keyClass = Class.forName("net.minecraft.resources.ResourceLocation");',
            '                Class<?> resourceKeyClass = Class.forName("net.minecraft.resources.ResourceKey");',
            '                Object registryKey = Class.forName("net.minecraft.core.registries.Registries").getField("RECIPE").get(null);',
            '                java.lang.reflect.Method keyFactory = keyClass.getMethod("fromNamespaceAndPath", String.class, String.class);',
            '                java.lang.reflect.Method resourceKeyFactory = resourceKeyClass.getMethod("create", resourceKeyClass, keyClass);',
            '                java.lang.reflect.Method findRecipe = recipeAccess.getClass().getMethod("byKey", resourceKeyClass);',
        ))
        for name in manifest["recipes"]:
            lines.extend((
                "                {",
                f'                    Object key = keyFactory.invoke(null, "{mod_id}", "{name}");',
                "                    Object typedKey = resourceKeyFactory.invoke(null, registryKey, key);",
                "                    Object recipe = findRecipe.invoke(recipeAccess, typedKey);",
                "                    if (!(recipe instanceof java.util.Optional<?> optional) || optional.isEmpty()) {",
                f'                        throw new AssertionError("GameTest RecipeManager did not load actual recipe: {mod_id}:{name}");',
                "                    }",
                "                }",
            ))
        lines.append("            }")
    lines.extend((
        "        } catch (ReflectiveOperationException | ClassCastException error) {",
        '            throw new AssertionError("GameTest target runtime API is incompatible", error);',
        "        }",
        f"        {_END}",
    ))
    return "\n".join(lines)


def eligible_for_managed_runtime(approved: Any) -> bool:
    """May plan a content GameTest; full eligibility checks follow generation."""
    return bool(_eligible_modules(approved))


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
        "recipes": list(manifest["recipes"]),
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
