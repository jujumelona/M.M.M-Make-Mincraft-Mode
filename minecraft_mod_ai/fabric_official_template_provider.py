from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


class FabricTemplateProviderError(RuntimeError):
    pass


class FabricTemplateSafetyError(FabricTemplateProviderError):
    """An unsafe scaffold cannot be repaired by rebinding toolchain versions."""


_FABRIC_CLI = "https://fabricmc.net/cli"
_DENO_LATEST = "https://dl.deno.land/release-latest.txt"
_DENO_RELEASE = "https://dl.deno.land/release/{version}/{asset}"


def bootstrap_fabric_project(
    *,
    project_root: str | Path,
    spec: Any,
    adapter: Any,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Generate a clean project using Fabric's maintained official template CLI.

    MMM deliberately does not carry a future-Minecraft Gradle/template fork. The
    selected game version is passed to Fabric's own generator, then all later source
    implementation is performed by the central-AI/compile-repair path.
    """

    root = Path(project_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FabricTemplateProviderError(
            f"Fabric official template target must be empty: {root}"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_root).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    deno = _ensure_deno(cache)

    command = [
        str(deno),
        "run",
        "-A",
        _FABRIC_CLI,
        "init",
        str(root),
        "-n",
        str(spec.mod_name),
        "-m",
        str(spec.mod_id),
        "-p",
        str(spec.package_name),
        "-v",
        str(adapter.minecraft_version),
        # Supplying one advanced option keeps the official CLI fully non-interactive.
        # Dynamic live targets standardize on Mojang names; 26.1+ is unobfuscated and
        # the option becomes effectively redundant there.
        "-o",
        "mojangMappings",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(os.environ.get("MMM_FABRIC_TEMPLATE_TIMEOUT", "240")),
        check=False,
    )
    if completed.returncode != 0:
        raise FabricTemplateProviderError(
            "Fabric official CLI failed for the selected target.\n"
            + (completed.stdout or "")[-8000:]
        )
    if not root.is_dir():
        raise FabricTemplateProviderError(
            "Fabric official CLI reported success without creating the project."
        )

    properties = _read_properties(root / "gradle.properties")
    actual_mc = properties.get("minecraft_version", "")
    if actual_mc != adapter.minecraft_version:
        raise FabricTemplateProviderError(
            "Fabric official template generated a different Minecraft target: "
            f"expected={adapter.minecraft_version}, actual={actual_mc!r}"
        )
    actual_java = _java_release(root)
    if actual_java != str(adapter.java_version):
        raise FabricTemplateProviderError(
            "Fabric official template Java target changed after target discovery; "
            f"expected={adapter.java_version}, actual={actual_java!r}. Restart planning."
        )

    provider_defaults = {
        "minecraft_version": actual_mc,
        "loader_version": properties.get("loader_version", ""),
        "fabric_api": properties.get("fabric_version", "")
        or properties.get("fabric_api_version", ""),
        "loom": properties.get("loom_version", ""),
        "gradle": _gradle_wrapper_version(root),
        "java": actual_java,
    }
    _pin_generated_toolchain(root, adapter)
    _clean_fresh_template_examples(root, spec)
    runtime_contract = _install_host_runtime_contract(root, spec, adapter)
    gametest_contract = _install_host_gametest_contract(root, spec)

    properties = _read_properties(root / "gradle.properties")
    actual_loader = properties.get("loader_version", "")
    actual_api = properties.get("fabric_version", "") or properties.get(
        "fabric_api_version", ""
    )
    actual_loom = properties.get("loom_version", "")
    actual_gradle = _gradle_wrapper_version(root)
    mismatches = {
        key: (actual, expected)
        for key, actual, expected in (
            ("loader", actual_loader, adapter.fabric_loader),
            ("fabric_api", actual_api, adapter.fabric_api),
            ("loom", actual_loom, adapter.fabric_loom),
            ("gradle", actual_gradle, adapter.gradle),
        )
        if actual != expected
    }
    if mismatches:
        raise FabricTemplateProviderError(
            "Pinned Fabric scaffold still disagrees with the approved target receipt: "
            + ", ".join(
                f"{key} expected={expected!r} actual={actual!r}"
                for key, (actual, expected) in sorted(mismatches.items())
            )
        )

    receipt = {
        "schema_version": "mmm/fabric-official-template-v2",
        "provider": "fabricmc.net/cli",
        "provider_url": _FABRIC_CLI,
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "loader_version": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "java": adapter.java_version,
        "mappings": "mojang",
        "deno": _deno_version(deno),
        "provider_defaults_before_pin": provider_defaults,
        "verified_generated_toolchain": {
            "minecraft_version": actual_mc,
            "loader_version": actual_loader,
            "fabric_api": actual_api,
            "loom": actual_loom,
            "gradle": actual_gradle,
            "java": actual_java,
        },
        "runtime_contract": runtime_contract,
        "gametest_contract": gametest_contract,
        "project_manifest_sha256": _manifest_hash(root),
    }
    _write_platform_lock(root, adapter, receipt)
    return receipt


def _ensure_deno(cache_root: Path) -> Path:
    configured = os.environ.get("MMM_DENO_CMD", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path.resolve()
        found = shutil.which(configured)
        if found:
            return Path(found).resolve()
        raise FabricTemplateProviderError(f"MMM_DENO_CMD is not executable: {configured}")

    found = shutil.which("deno")
    if found:
        return Path(found).resolve()

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise FabricTemplateProviderError(
            "Deno is required for Fabric's official template provider on this host. "
            "Install Deno or set MMM_DENO_CMD."
        )
    if machine in {"x86_64", "amd64"}:
        target = "x86_64-unknown-linux-gnu"
    elif machine in {"aarch64", "arm64"}:
        target = "aarch64-unknown-linux-gnu"
    else:
        raise FabricTemplateProviderError(
            f"Automatic Deno bootstrap does not support architecture {machine!r}."
        )

    version = _download_text(_DENO_LATEST).strip()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[-+][A-Za-z0-9._-]+)?", version):
        raise FabricTemplateProviderError(
            f"Deno latest-release endpoint returned an invalid version: {version!r}"
        )
    install_dir = cache_root / "deno" / version
    binary = install_dir / "deno"
    if binary.is_file():
        binary.chmod(0o755)
        return binary

    asset = f"deno-{target}.zip"
    archive = install_dir / asset
    install_dir.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(_download_bytes(_DENO_RELEASE.format(version=version, asset=asset)))
    checksum_text = _download_text(
        _DENO_RELEASE.format(version=version, asset=asset + ".sha256sum")
    )
    expected = checksum_text.strip().split()[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise FabricTemplateProviderError("Deno release checksum was invalid.")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise FabricTemplateProviderError("Downloaded Deno release failed SHA-256 verification.")
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if names != ["deno"]:
                raise FabricTemplateProviderError(
                    f"Unexpected Deno release archive members: {names[:8]}"
                )
            bundle.extract("deno", install_dir)
    except zipfile.BadZipFile as exc:
        raise FabricTemplateProviderError("Downloaded Deno archive is invalid.") from exc
    binary.chmod(0o755)
    archive.unlink(missing_ok=True)
    return binary


def _download_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "MMM-fabric-provider/1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.read()
    except Exception as exc:  # pragma: no cover - network-specific
        raise FabricTemplateProviderError(f"Failed to download official bootstrap input: {url}: {exc}") from exc


def _download_text(url: str) -> str:
    try:
        return _download_bytes(url).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FabricTemplateProviderError(f"Official bootstrap text was not UTF-8: {url}") from exc


def _pin_generated_toolchain(root: Path, adapter: Any) -> None:
    """Pin volatile official-template defaults to the approved immutable receipt."""

    properties_path = root / "gradle.properties"
    if not properties_path.is_file() or properties_path.is_symlink():
        raise FabricTemplateProviderError("Fabric template omitted gradle.properties.")

    raw_lines = properties_path.read_text(encoding="utf-8").splitlines()
    present_keys = {
        line.split("=", 1)[0].strip()
        for line in raw_lines
        if "=" in line and not line.lstrip().startswith("#")
    }
    api_key = (
        "fabric_api_version"
        if "fabric_api_version" in present_keys
        else "fabric_version"
        if "fabric_version" in present_keys
        else ""
    )
    if not api_key:
        raise FabricTemplateProviderError(
            "Fabric template exposes no recognized Fabric API version property."
        )

    replacements = {
        "loader_version": str(adapter.fabric_loader),
        "loom_version": str(adapter.fabric_loom),
        api_key: str(adapter.fabric_api),
    }
    missing = set(replacements) - present_keys
    if missing:
        raise FabricTemplateProviderError(
            "Fabric template omitted required dependency properties: "
            + ", ".join(sorted(missing))
        )

    rewritten: list[str] = []
    for raw in raw_lines:
        if "=" not in raw or raw.lstrip().startswith("#"):
            rewritten.append(raw)
            continue
        key, _value = raw.split("=", 1)
        normalized = key.strip()
        if normalized in replacements:
            rewritten.append(f"{key[: len(key) - len(key.lstrip())]}{normalized}={replacements[normalized]}")
        else:
            rewritten.append(raw)
    properties_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    wrapper = root / "gradle/wrapper/gradle-wrapper.properties"
    if not wrapper.is_file() or wrapper.is_symlink():
        raise FabricTemplateProviderError("Fabric template omitted the Gradle wrapper properties.")
    wrapper_lines = wrapper.read_text(encoding="utf-8").splitlines()
    distribution_seen = False
    checksum_seen = False
    pinned_lines: list[str] = []
    for raw in wrapper_lines:
        if raw.startswith("distributionUrl="):
            distribution_seen = True
            pinned_lines.append(
                "distributionUrl=https\\://services.gradle.org/distributions/"
                f"gradle-{adapter.gradle}-bin.zip"
            )
            continue
        if raw.startswith("distributionSha256Sum="):
            checksum_seen = True
            pinned_lines.append(f"distributionSha256Sum={adapter.gradle_sha256}")
            continue
        pinned_lines.append(raw)
    if not distribution_seen:
        raise FabricTemplateProviderError("Fabric template Gradle wrapper has no distributionUrl.")
    if not checksum_seen:
        pinned_lines.append(f"distributionSha256Sum={adapter.gradle_sha256}")
    wrapper.write_text("\n".join(pinned_lines) + "\n", encoding="utf-8")


def _clean_fresh_template_examples(root: Path, spec: Any) -> None:
    """Remove CLI demonstration code only during bootstrap of an empty project.

    Never call this on imported or already implemented projects. At this point
    every Java source is owned by the official scaffold, before MMM generation.
    Keep the upstream license and wrapper/build infrastructure intact.
    """
    root = root.resolve()

    def require_owned(path: Path) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise FabricTemplateSafetyError(f"Fabric template path escaped the project: {path}")

    resources = root / "src/main/resources"
    metadata_path = resources / "fabric.mod.json"
    readme = root / "README.md"
    require_owned(metadata_path)
    require_owned(readme)
    require_owned(root / "src")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FabricTemplateProviderError("Fabric template metadata is invalid.") from exc
    if not isinstance(metadata, dict):
        raise FabricTemplateProviderError("Fabric template metadata must be an object.")
    removals: list[Path] = []
    for entry in metadata.get("mixins", []):
        name = entry.get("config") if isinstance(entry, dict) else entry
        if not isinstance(name, str):
            raise FabricTemplateProviderError("Fabric template mixin config is invalid.")
        for resource_root in (resources, root / "src/client/resources"):
            require_owned(resource_root)
            path = resource_root / name
            require_owned(path)
            if not path.resolve().is_relative_to(resource_root.resolve()) or path.is_symlink():
                raise FabricTemplateSafetyError("Fabric template mixin config escaped resources.")
            if path.is_file():
                removals.append(path)
    for path in (root / "src").rglob("*.java"):
        require_owned(path)
        removals.append(path)
    # Validate every destination before removing even a single scaffold file.
    for path in dict.fromkeys(removals):
        path.unlink()
    metadata["entrypoints"] = {}
    metadata.pop("mixins", None)
    metadata["description"] = str(spec.summary)
    metadata["authors"] = []
    metadata["contact"] = {}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme.write_text(
        f"# {spec.mod_name}\n\n{spec.summary}\n\n"
        "Build with `./gradlew build` (Windows: `gradlew.bat build`).\n"
        "The Gradle wrapper and gradle.properties pin the target toolchain.\n\n"
        "See LICENSE for the scaffold license.\n",
        encoding="utf-8",
    )


def _install_host_runtime_contract(
    root: Path,
    spec: Any,
    adapter: Any,
) -> dict[str, Any]:
    """Normalize the official scaffold to MMM's immutable runtime/JAR contract."""

    metadata_path = root / "src/main/resources/fabric.mod.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise FabricTemplateProviderError(
            "Fabric official template omitted src/main/resources/fabric.mod.json."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FabricTemplateProviderError(
            "Fabric official template fabric.mod.json is invalid."
        ) from exc
    if not isinstance(metadata, dict):
        raise FabricTemplateProviderError(
            "Fabric official template fabric.mod.json must be an object."
        )

    expected_depends = {
        "fabricloader": str(adapter.fabric_loader),
        "minecraft": str(adapter.minecraft_version),
        "java": str(adapter.java_version),
        "fabric-api": str(adapter.fabric_api),
    }
    depends = metadata.setdefault("depends", {})
    if not isinstance(depends, dict):
        raise FabricTemplateProviderError(
            "Fabric official template depends must be an object."
        )
    depends.update(expected_depends)
    metadata["environment"] = "*"

    main_class = "".join(
        part.capitalize() for part in str(spec.mod_id).split("_")
    ) + "Mod"
    main_entrypoint = f"{spec.package_name}.{main_class}"
    entrypoints = metadata.setdefault("entrypoints", {})
    if not isinstance(entrypoints, dict):
        raise FabricTemplateProviderError(
            "Fabric official template entrypoints must be an object."
        )
    main_entrypoints = entrypoints.setdefault("main", [])
    if not isinstance(main_entrypoints, list):
        raise FabricTemplateProviderError(
            "Fabric official template main entrypoint must be a list."
        )

    def entrypoint_value(item: Any) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, dict) and isinstance(item.get("value"), str):
            return str(item["value"])
        return None

    if main_entrypoint not in {
        value
        for item in main_entrypoints
        if (value := entrypoint_value(item)) is not None
    }:
        main_entrypoints.append(main_entrypoint)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_path = (
        root
        / "src/main/java"
        / Path(*str(spec.package_name).split("."))
        / f"{main_class}.java"
    )
    if source_path.is_symlink():
        raise FabricTemplateProviderError(
            "Canonical Fabric main entrypoint source must not be a symlink."
        )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        source_path.write_text(
            f"""package {spec.package_name};

import net.fabricmc.api.ModInitializer;

public final class {main_class} implements ModInitializer {{
    @Override
    public void onInitialize() {{
        // Host-owned baseline. Generated feature binders may append calls here.
    }}
}}
""",
            encoding="utf-8",
        )
    elif not source_path.is_file():
        raise FabricTemplateProviderError(
            "Canonical Fabric main entrypoint source is not a regular file."
        )

    lang_resources: list[str] = []
    for locale in ("en_us", "ko_kr"):
        path = (
            root
            / "src/main/resources/assets"
            / str(spec.mod_id)
            / "lang"
            / f"{locale}.json"
        )
        if path.is_symlink():
            raise FabricTemplateProviderError(
                f"Canonical language resource must not be a symlink: {locale}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")
        elif not path.is_file():
            raise FabricTemplateProviderError(
                f"Canonical language resource is not a regular file: {locale}"
            )
        lang_resources.append(path.relative_to(root).as_posix())

    return {
        "main_entrypoint": main_entrypoint,
        "main_source": source_path.relative_to(root).as_posix(),
        "lang_resources": lang_resources,
        "depends": expected_depends,
    }


def _install_host_gametest_contract(root: Path, spec: Any) -> dict[str, str]:
    """Install Fabric GameTest in the dedicated Loom gametest source set."""

    build_path = root / "build.gradle"
    if not build_path.is_file() or build_path.is_symlink():
        raise FabricTemplateProviderError(
            "Fabric official template omitted the Groovy build.gradle required by "
            "the host GameTest contract."
        )

    test_mod_id = f"{spec.mod_id}_gametest"
    build_text = build_path.read_text(encoding="utf-8", errors="strict")
    additions: list[str] = []
    if "configureTests" not in build_text:
        additions.append(
            f"""// M.M.M host-owned server GameTest contract
fabricApi {{
    configureTests {{
        createSourceSet = true
        modId = "{test_mod_id}"
        enableGameTests = true
        enableClientGameTests = false
    }}
}}
"""
        )
    if "fabric-api.gametest.report-file" not in build_text:
        additions.append(
            """// M.M.M structured GameTest evidence
// Bind the property on Loom's generated gameTest run so runGameTest inherits it.
loom {
    runs {
        gameTest {
            property "fabric-api.gametest.report-file", file('build/gametest-report.xml').absolutePath
        }
    }
}
"""
        )
    if additions:
        build_path.write_text(
            build_text.rstrip() + "\n\n" + "\n".join(additions).rstrip() + "\n",
            encoding="utf-8",
        )

    main_class = "".join(part.capitalize() for part in str(spec.mod_id).split("_")) + "Mod"
    gametest_class = main_class + "GameTests"
    gametest_entrypoint = f"{spec.package_name}.{gametest_class}"

    main_metadata_path = root / "src/main/resources/fabric.mod.json"
    if not main_metadata_path.is_file() or main_metadata_path.is_symlink():
        raise FabricTemplateProviderError(
            "Fabric official template omitted src/main/resources/fabric.mod.json."
        )
    try:
        main_metadata = json.loads(main_metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FabricTemplateProviderError(
            "Fabric official template fabric.mod.json is invalid."
        ) from exc
    if not isinstance(main_metadata, dict):
        raise FabricTemplateProviderError(
            "Fabric official template fabric.mod.json must be an object."
        )
    main_entrypoints = main_metadata.get("entrypoints")
    main_changed = False
    if isinstance(main_entrypoints, dict):
        existing = main_entrypoints.get("fabric-gametest")
        if isinstance(existing, list) and gametest_entrypoint in existing:
            filtered = [item for item in existing if item != gametest_entrypoint]
            if filtered:
                main_entrypoints["fabric-gametest"] = filtered
            else:
                main_entrypoints.pop("fabric-gametest", None)
            main_changed = True
    if main_changed:
        main_metadata_path.write_text(
            json.dumps(main_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    old_source = (
        root
        / "src/main/java"
        / Path(*str(spec.package_name).split("."))
        / f"{gametest_class}.java"
    )
    if old_source.is_symlink():
        raise FabricTemplateProviderError(
            "Legacy host GameTest source must not be a symlink."
        )
    if old_source.is_file():
        old_source.unlink()

    gametest_metadata_path = root / "src/gametest/resources/fabric.mod.json"
    gametest_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    gametest_metadata = {
        "schemaVersion": 1,
        "id": test_mod_id,
        "version": "1.0.0",
        "name": f"{spec.mod_id} GameTests",
        "environment": "*",
        "entrypoints": {"fabric-gametest": [gametest_entrypoint]},
        "depends": {str(spec.mod_id): "*"},
    }
    gametest_metadata_path.write_text(
        json.dumps(gametest_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_path = (
        root
        / "src/gametest/java"
        / Path(*str(spec.package_name).split("."))
        / f"{gametest_class}.java"
    )
    if source_path.is_symlink():
        raise FabricTemplateProviderError(
            "Canonical Fabric GameTest source must not be a symlink."
        )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"""package {spec.package_name};

import net.fabricmc.fabric.api.gametest.v1.FabricGameTest;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.gametest.framework.GameTest;
import net.minecraft.gametest.framework.GameTestHelper;

public final class {gametest_class} implements FabricGameTest {{
    @GameTest(template = FabricGameTest.EMPTY_STRUCTURE)
    public void generatedRegistriesAreLive(GameTestHelper context) {{
        if (!FabricLoader.getInstance().isModLoaded("{spec.mod_id}")) {{
            throw new AssertionError("generated mod was not loaded by Fabric");
        }}
        context.succeed();
    }}
}}
""",
        encoding="utf-8",
    )

    return {
        "task": "runGameTest",
        "report": "build/gametest-report.xml",
        "entrypoint": gametest_entrypoint,
        "source": source_path.relative_to(root).as_posix(),
        "metadata": gametest_metadata_path.relative_to(root).as_posix(),
        "mod_id": test_mod_id,
    }


def _read_properties(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FabricTemplateProviderError(f"Fabric template omitted {path.name}.")
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _gradle_wrapper_version(root: Path) -> str:
    path = root / "gradle/wrapper/gradle-wrapper.properties"
    if not path.is_file() or path.is_symlink():
        raise FabricTemplateProviderError("Fabric template omitted the Gradle wrapper properties.")
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"gradle-([0-9][0-9A-Za-z_.-]*)-bin\.zip", text)
    if match is None:
        raise FabricTemplateProviderError("Could not determine generated Gradle wrapper version.")
    return match.group(1)


def _java_release(root: Path) -> str:
    candidates = (root / "build.gradle", root / "build.gradle.kts")
    for path in candidates:
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        patterns = (
            r"options\.release\s*=\s*(\d+)",
            r"options\.release\.set\((\d+)\)",
            r"JavaVersion\.VERSION_(\d+)",
            r"JavaLanguageVersion\.of\((\d+)\)",
        )
        for pattern_value in patterns:
            match = re.search(pattern_value, text)
            if match is not None:
                return match.group(1)
    raise FabricTemplateProviderError("Could not determine generated Java target release.")


def _deno_version(command: Path) -> str:
    completed = subprocess.run(
        [str(command), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    first = (completed.stdout or "").splitlines()
    return first[0].strip() if first else "unknown"


def _manifest_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and ".gradle" not in item.parts):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _write_platform_lock(root: Path, adapter: Any, receipt: dict[str, Any]) -> None:
    from .platform_generation_contract import write_platform_lock

    write_platform_lock(root, adapter, bootstrap=receipt)
