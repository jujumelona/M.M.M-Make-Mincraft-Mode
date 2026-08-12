from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any


_GITHUB_RELEASES = (
    "https://api.github.com/repos/chapmanjw/"
    "minecraft-java-fabric-mcp-server/releases?per_page=10"
)
_FABRIC_MAVEN = "https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api"
_USER_AGENT = "M.M.M-runtime-helper/1"


def install(runtime_manager_module: Any) -> None:
    cls = runtime_manager_module.MinecraftRuntimeManager
    original_prepare = cls.prepare_instance
    if getattr(original_prepare, "_mmm_external_mcp_helpers", False):
        return

    @wraps(original_prepare)
    def prepare_instance(self: Any, *args: Any, **kwargs: Any):
        result = original_prepare(self, *args, **kwargs)
        root = Path(str(result["instance_root"])).resolve()
        adapter = getattr(self, "_mmm_platform_adapter", None)
        receipt = _stage_helpers(self.workspace_root, root, adapter)
        result = dict(result)
        result["external_mcp_runtime_helpers"] = receipt
        return result

    prepare_instance._mmm_external_mcp_helpers = True
    cls.prepare_instance = prepare_instance


def _stage_helpers(workspace_root: Path, instance_root: Path, adapter: Any) -> dict[str, Any]:
    if adapter is None or str(getattr(adapter, "loader", "")) != "fabric":
        return _receipt("SKIPPED_UNSUPPORTED_LOADER", adapter, [])
    if os.environ.get("MMM_DISABLE_RUNTIME_MCP_HELPERS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _receipt("SKIPPED_BY_HOST_POLICY", adapter, [])

    cache = (Path(workspace_root) / ".cache/runtime-mcp-helpers").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    server_mods = instance_root / "mods"
    client_mods = instance_root / "client/mods"
    server_mods.mkdir(parents=True, exist_ok=True)
    client_mods.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []

    # Fabric API is a dependency of the in-game MCP helper and is also normally a
    # dependency of generated Fabric mods. Use the already-approved exact coordinate;
    # never choose a newer API independently of PlatformLock.
    fabric_api = str(getattr(adapter, "fabric_api", "")).strip()
    if fabric_api:
        try:
            api_artifact = _fabric_api_artifact(cache, fabric_api)
            _copy_to_runtime(api_artifact["path"], server_mods, client_mods)
            artifacts.append({**api_artifact, "role": "fabric_api"})
        except Exception as exc:
            artifacts.append(
                {
                    "role": "fabric_api",
                    "status": "UNAVAILABLE",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    try:
        helper = _fabric_mcp_helper(
            cache,
            minecraft_version=str(adapter.minecraft_version),
        )
    except Exception as exc:
        artifacts.append(
            {
                "role": "fabric_game_mcp",
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return _receipt("PARTIAL" if artifacts else "UNAVAILABLE", adapter, artifacts)

    if helper is None:
        artifacts.append(
            {
                "role": "fabric_game_mcp",
                "status": "NO_EXACT_TARGET_ASSET",
                "minecraft_version": str(adapter.minecraft_version),
            }
        )
        return _receipt("PARTIAL", adapter, artifacts)

    _copy_to_runtime(helper["path"], server_mods, client_mods)
    artifacts.append({**helper, "role": "fabric_game_mcp"})
    return _receipt("PASS", adapter, artifacts)


def _fabric_mcp_helper(cache: Path, *, minecraft_version: str) -> dict[str, Any] | None:
    releases = _json_request(_GITHUB_RELEASES)
    if not isinstance(releases, list):
        raise RuntimeError("GitHub releases response is not a list")
    suffix = f"+{minecraft_version}.jar"
    for release in releases:
        if not isinstance(release, dict) or release.get("draft"):
            continue
        for asset in release.get("assets", []):
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", ""))
            if not name.startswith("minecraft-fabric-mcp-") or not name.endswith(suffix):
                continue
            url = str(asset.get("browser_download_url", ""))
            digest = str(asset.get("digest", ""))
            if not url.startswith("https://github.com/"):
                raise RuntimeError("Runtime MCP helper asset URL is not reviewed HTTPS GitHub")
            if not digest.startswith("sha256:") or len(digest) != 71:
                raise RuntimeError("Runtime MCP helper release asset has no SHA-256 digest")
            target = cache / name
            _download_verified(url, target, digest.split(":", 1)[1])
            _validate_mod_jar(target, expected_mod_id_fragment="mcp")
            return {
                "status": "STAGED",
                "path": str(target),
                "filename": name,
                "sha256": digest,
                "release_tag": str(release.get("tag_name", "")),
                "minecraft_version": minecraft_version,
                "source": "chapmanjw/minecraft-java-fabric-mcp-server",
            }
    return None


def _fabric_api_artifact(cache: Path, version: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(version, safe="")
    filename = f"fabric-api-{version}.jar"
    url = f"{_FABRIC_MAVEN}/{encoded}/{urllib.parse.quote(filename, safe='+-._') }"
    target = cache / filename
    if not target.is_file():
        _download(url, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    _validate_mod_jar(target, expected_mod_id_fragment="fabric")
    return {
        "status": "STAGED",
        "path": str(target),
        "filename": filename,
        "sha256": "sha256:" + digest,
        "version": version,
        "source": "maven.fabricmc.net",
    }


def _copy_to_runtime(path_value: str, server_mods: Path, client_mods: Path) -> None:
    path = Path(path_value).resolve()
    for directory in (server_mods, client_mods):
        target = directory / path.name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(path.read_bytes()).digest():
                raise RuntimeError(f"Runtime helper collision for {target.name}")
            continue
        shutil.copy2(path, target)


def _download_verified(url: str, target: Path, expected_sha256: str) -> None:
    if target.is_file():
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual == expected_sha256:
            return
        target.unlink()
    _download(url, target)
    actual = hashlib.sha256(target.read_bytes()).hexdigest()
    if actual != expected_sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"Runtime MCP helper SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers=_headers())
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _json_request(url: str) -> Any:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _headers() -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_mod_jar(path: Path, *, expected_mod_id_fragment: str) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Downloaded runtime helper is not a JAR/ZIP: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "fabric.mod.json" not in names:
            raise RuntimeError(f"Downloaded runtime helper lacks fabric.mod.json: {path.name}")
        metadata = json.loads(archive.read("fabric.mod.json").decode("utf-8"))
    mod_id = str(metadata.get("id", "")).casefold()
    if expected_mod_id_fragment.casefold() not in mod_id:
        raise RuntimeError(
            f"Unexpected Fabric mod id {mod_id!r} in runtime helper {path.name}"
        )


def _receipt(status: str, adapter: Any, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    target = {
        "minecraft_version": str(getattr(adapter, "minecraft_version", "")),
        "loader": str(getattr(adapter, "loader", "")),
        "fabric_api": str(getattr(adapter, "fabric_api", "")),
    }
    payload = {
        "schema_version": "mmm/runtime-mcp-helper-staging-v1",
        "status": status,
        "target": target,
        "test_harness_only": True,
        "packaged_into_release": False,
        "artifacts": artifacts,
    }
    payload["receipt_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload
