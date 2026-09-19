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
