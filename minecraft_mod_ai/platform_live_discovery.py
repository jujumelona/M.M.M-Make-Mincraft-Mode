from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


class PlatformDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveFabricTarget:
    minecraft_version: str
    stable: bool
    loader_version: str
    fabric_api_version: str
    loom_version: str
    java_version: str
    gradle_version: str
    gradle_sha256: str
    mappings_kind: str
    mappings_version: str
    discovery_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "minecraft_version": self.minecraft_version,
            "stable": self.stable,
            "loader_version": self.loader_version,
            "fabric_api_version": self.fabric_api_version,
            "loom_version": self.loom_version,
            "java_version": self.java_version,
            "gradle_version": self.gradle_version,
            "gradle_sha256": self.gradle_sha256,
            "mappings_kind": self.mappings_kind,
            "mappings_version": self.mappings_version,
            "discovery_sha256": self.discovery_sha256,
        }


_META = "https://meta.fabricmc.net"
_MAVEN = "https://maven.fabricmc.net"
_FABRIC_DEVELOP = "https://fabricmc.net/develop/"
_FABRIC_WRAPPER = (
    "https://raw.githubusercontent.com/FabricMC/fabricmc.net/main/"
    "scripts/src/lib/template/templates/gradle/wrapper/gradle/wrapper/"
    "gradle-wrapper.properties"
)
_MOJANG_MANIFEST = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json"


def _fetch(url: str, *, timeout: int = 20) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MMM-platform-discovery/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:  # pragma: no cover - network-specific
        raise PlatformDiscoveryError(f"official platform discovery failed: {url}: {exc}") from exc


def _json(url: str) -> Any:
    try:
        return json.loads(_fetch(url).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlatformDiscoveryError(f"official JSON response was invalid: {url}") from exc


def _maven_versions(path: str) -> list[str]:
    raw = _fetch(_MAVEN + path)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise PlatformDiscoveryError(f"invalid Fabric Maven metadata: {path}") from exc
    return [
        str(node.text).strip()
        for node in root.findall("./versioning/versions/version")
        if node.text and str(node.text).strip()
    ]


@lru_cache(maxsize=1)
def discover_game_versions() -> tuple[dict[str, Any], ...]:
    payload = _json(_META + "/v2/versions/game")
    if not isinstance(payload, list):
        raise PlatformDiscoveryError("Fabric Meta game-version response was not a list")
    result: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        version = str(row.get("version", "")).strip()
        if version:
            result.append({"version": version, "stable": bool(row.get("stable"))})
    if not result:
        raise PlatformDiscoveryError("Fabric Meta returned no Minecraft versions")
    return tuple(result)


def latest_stable_versions(limit: int = 6) -> tuple[str, ...]:
    stable = [row["version"] for row in discover_game_versions() if row["stable"]]
    return tuple(stable[: max(1, int(limit))])


def _stable_loader() -> str:
    payload = _json(_META + "/v2/versions/loader")
    if not isinstance(payload, list):
        raise PlatformDiscoveryError("Fabric loader response was not a list")
    for row in payload:
        if isinstance(row, dict) and row.get("stable") and row.get("version"):
            return str(row["version"])
    raise PlatformDiscoveryError("Fabric Meta returned no stable loader")


def _yarn_for(version: str) -> str:
    payload = _json(_META + "/v2/versions/yarn/" + version + "?limit=1")
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and first.get("version"):
            return str(first["version"])
    return ""


def _api_for(version: str) -> str:
    versions = _maven_versions("/net/fabricmc/fabric-api/fabric-api/maven-metadata.xml")
    exact_suffixes = ("+" + version, "-" + version)
    matches = [value for value in versions if value.endswith(exact_suffixes)]
    if matches:
        return matches[-1]
    # Modern calendar-version Minecraft releases use their release number as the
    # Fabric API branch. Future releases therefore require no MMM version table.
    release = version.split("-", 1)[0]
    major = release.split(".", 1)[0]
    if major.isdigit() and int(major) >= 26:
        matches = [
            value
            for value in versions
            if value.endswith(("+" + release, "-" + release))
        ]
        if matches:
            return matches[-1]
    raise PlatformDiscoveryError(
        f"Fabric API has no artifact discoverable for Minecraft {version}"
    )


def _loom_version() -> str:
    text = _fetch(_FABRIC_DEVELOP).decode("utf-8", errors="replace")
    match = re.search(r"loom_version=([^\s<]+)", text)
    if not match:
        match = re.search(r"recommended loom version is\s*<strong>([^<]+)", text, re.I)
    if not match:
        raise PlatformDiscoveryError("Fabric develop page exposed no recommended Loom version")
    return match.group(1).strip()


def _gradle_version() -> str:
    text = _fetch(_FABRIC_WRAPPER).decode("utf-8", errors="replace")
    match = re.search(r"gradle-([0-9][0-9A-Za-z_.-]*)-bin\.zip", text)
    if not match:
        raise PlatformDiscoveryError("Fabric template exposed no Gradle wrapper version")
    return match.group(1)


def _gradle_sha256(version: str) -> str:
    raw = _fetch(
        f"https://services.gradle.org/distributions/gradle-{version}-bin.zip.sha256"
    ).decode("ascii", errors="ignore").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", raw):
        raise PlatformDiscoveryError("Gradle distribution checksum was invalid")
    return raw


def _mojang_java_version(version: str) -> str:
    manifest = _json(_MOJANG_MANIFEST)
    rows = manifest.get("versions", []) if isinstance(manifest, dict) else []
    target_url = ""
    for row in rows:
        if isinstance(row, dict) and str(row.get("id", "")) == version:
            target_url = str(row.get("url", ""))
            break
    if not target_url:
        raise PlatformDiscoveryError(
            f"Mojang version manifest does not contain Minecraft {version}"
        )
    detail = _json(target_url)
    java = detail.get("javaVersion", {}) if isinstance(detail, dict) else {}
    major = java.get("majorVersion") if isinstance(java, dict) else None
    if not isinstance(major, int) or major <= 0:
        raise PlatformDiscoveryError(
            f"Mojang metadata exposed no Java major version for Minecraft {version}"
        )
    return str(major)


@lru_cache(maxsize=32)
def discover_fabric_target(version: str) -> LiveFabricTarget:
    version = str(version).strip()
    rows = discover_game_versions()
    row = next((item for item in rows if item["version"] == version), None)
    if row is None:
        raise PlatformDiscoveryError(
            f"Minecraft {version} is not advertised by the official Fabric Meta API"
        )

    loader = _stable_loader()
    api = _api_for(version)
    loom = _loom_version()
    gradle = _gradle_version()
    java = _mojang_java_version(version)
    yarn = _yarn_for(version)
    mappings_kind = "yarn" if yarn else "mojang"
    mappings_version = yarn or "mojang"
    payload = {
        "source": "official-live-discovery-v1",
        "minecraft_version": version,
        "stable": bool(row["stable"]),
        "loader_version": loader,
        "fabric_api_version": api,
        "loom_version": loom,
        "java_version": java,
        "gradle_version": gradle,
        "gradle_sha256": _gradle_sha256(gradle),
        "mappings_kind": mappings_kind,
        "mappings_version": mappings_version,
        "sources": [
            _META,
            _MAVEN,
            _FABRIC_DEVELOP,
            _FABRIC_WRAPPER,
            _MOJANG_MANIFEST,
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LiveFabricTarget(
        minecraft_version=version,
        stable=bool(row["stable"]),
        loader_version=loader,
        fabric_api_version=api,
        loom_version=loom,
        java_version=java,
        gradle_version=gradle,
        gradle_sha256=payload["gradle_sha256"],
        mappings_kind=mappings_kind,
        mappings_version=mappings_version,
        discovery_sha256="sha256:" + digest,
    )
