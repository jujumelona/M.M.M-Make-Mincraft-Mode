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
    actual_loader = properties.get("loader_version", "")
    if actual_loader and actual_loader != adapter.fabric_loader:
        raise FabricTemplateProviderError(
            "Fabric official template loader changed after target discovery; "
            "restart planning so the user approves one immutable target receipt."
        )
    actual_api = properties.get("fabric_version", "") or properties.get(
        "fabric_api_version", ""
    )
    if actual_api and actual_api != adapter.fabric_api:
        raise FabricTemplateProviderError(
            "Fabric official template API changed after target discovery; "
            "restart planning so the user approves the new coordinates."
        )

    receipt = {
        "schema_version": "mmm/fabric-official-template-v1",
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
    target = root / ".minecraft_ai" / "platform-lock.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "mmm/generated-platform-lock-v2",
        "adapter_id": adapter.adapter_id,
        "edition": adapter.edition,
        "loader": adapter.loader,
        "minecraft_version": adapter.minecraft_version,
        "java_version": adapter.java_version,
        "yarn_mappings": "mojang",
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "gradle_sha256": adapter.gradle_sha256,
        "source_api_family": "fabric_live_ai",
        "bootstrap": receipt,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
