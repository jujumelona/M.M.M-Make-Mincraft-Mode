from __future__ import annotations

"""Provider-authoritative Gradle scaffolding for executable Minecraft targets."""

import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .platform_catalog import PlatformAdapter, adapter_for_target, executable_loaders

GRADLE_CACHE_RELEASE_TAG = "gradle-runtime-cache-v1-immutable"
GRADLE_CACHE_RELEASE_BASE = (
    "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode/releases/download/"
    + GRADLE_CACHE_RELEASE_TAG
)
# Artifact-integrity compatibility ledgers, never target-support authorities.
GRADLE_DISTRIBUTION_SHA256S = {
    "7.6": "7ba68c54029790ab444b39d7e293d3236b2632631fb5f2e012bb28b4ff669e4b",
    "8.1": "a62c5f99585dd9e1f95dab7b9415a0e698fa9dd1e6c38537faa81ac078f4d23e",
    "8.5": "9d926787066a081739e8200858338b4a69e837c3a821a33aca9db09dd4a41026",
    "8.8": "a4b4158601f8636cdeeab09bd76afb640030bb5b144aafe261a5e8af027dc612",
    "8.10.2": "31c55713e40233a8303827ceb42ca48a47267a0ad4bab9177123121e71524c26",
}
GRADLE_WRAPPER_SHA256S = {
    "7.6": "c5a643cf80162e665cc228f7b16f343fef868e47d3a4836f62e18b7e17ac018a",
    "8.1": "ed2c26eba7cfb93cc2b7785d05e534f07b5b48b5e7fc941921cd098628abca58",
    "8.5": "d3b261c2820e9e3d8d639ed084900f11f4a86050a8f83342ade7b6bc9b0d2bdd",
    "8.8": "cb0da6751c2b753a16ac168bb354870ebb1e162e9083f116729cec9c781156b8",
    "8.10.2": "2db75c40782f5e8ba1fc278a5574bab070adccb2d21ca5a6e5ed840888448046",
}
GRADLE_SOURCE_TAGS = {
    "7.6": "v7.6.0",
    "8.1": "v8.1.0",
    "8.5": "v8.5.0",
    "8.8": "v8.8.0",
    "8.10.2": "v8.10.2",
}
_FABRIC_WRAPPER_DIR = "scripts/src/lib/template/templates/gradle/wrapper/gradle/wrapper"
_FABRIC_RAW = "https://raw.githubusercontent.com/FabricMC/fabricmc.net/main/"
_FABRIC_API = "https://api.github.com/repos/FabricMC/fabricmc.net/contents/"


@dataclass(frozen=True)
class VerifiedScaffoldTemplate:
    loader: str
    minecraft_version: str
    gradle_version: str
    distribution_sha256: str
    build_gradle: str
    settings_gradle: str
    adapter_id: str = ""
    java_release: int = 0
    loom_plugin_id: str = ""
    distribution_url: str = ""


class UnsupportedTargetSpecificationError(ValueError):
    """The provider receipt cannot produce a safe executable scaffold."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode())
    digest.update(data)
    return digest.hexdigest()


def _cache_root() -> Path:
    override = os.environ.get("MMM_GRADLE_WRAPPER_CACHE_DIR", "").strip()
    return (
        Path(override).expanduser().resolve()
        if override
        else (Path.home() / ".cache" / "mmm" / "gradle-wrapper").resolve()
    )


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "MMM-scaffold/2"})


def _fetch_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(_request(url), timeout=timeout) as response:
        return response.read().decode()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(url), timeout=120) as response, path.open("wb") as out:
        shutil.copyfileobj(response, out)


def _validate_wrapper(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Gradle wrapper JAR is missing or unsafe")
    try:
        with zipfile.ZipFile(path) as archive:
            ok = "org/gradle/wrapper/GradleWrapperMain.class" in archive.namelist()
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Gradle wrapper JAR is invalid") from exc
    if not ok:
        raise RuntimeError("Gradle wrapper JAR lacks GradleWrapperMain")


@lru_cache(maxsize=8)
def _live_wrapper_pin(gradle: str) -> tuple[str, str, int]:
    properties = _fetch_text(_FABRIC_RAW + _FABRIC_WRAPPER_DIR + "/gradle-wrapper.properties")
    match = re.search(r"gradle-([0-9][0-9A-Za-z_.-]*)-bin\.zip", properties)
    if not match or match.group(1) != gradle:
        raise RuntimeError(f"Fabric wrapper does not match provider Gradle {gradle}")
    rows = json.loads(_fetch_text(_FABRIC_API + _FABRIC_WRAPPER_DIR + "?ref=main"))
    row = next((item for item in rows if item.get("name") == "gradle-wrapper.jar"), None)
    if not isinstance(row, dict):
        raise RuntimeError("Fabric template exposes no wrapper JAR")
    url = str(row.get("download_url") or "")
    sha = str(row.get("sha") or "").lower()
    size = row.get("size")
    if not url.startswith(_FABRIC_RAW) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("Fabric wrapper metadata failed origin/integrity validation")
    if type(size) is not int or size <= 0:
        raise RuntimeError("Fabric wrapper metadata has invalid size")
    return url, sha, size


def _ensure_wrapper(adapter: PlatformAdapter) -> Path:
    gradle = adapter.gradle
    target = _cache_root() / gradle / "gradle-wrapper.jar"
    known = GRADLE_WRAPPER_SHA256S.get(gradle)
    if known and target.is_file() and not target.is_symlink() and _sha256(target) == known:
        _validate_wrapper(target)
        return target
    if known:
        tag = GRADLE_SOURCE_TAGS[gradle]
        candidates = (
            f"{GRADLE_CACHE_RELEASE_BASE}/gradle-{gradle}-wrapper.jar",
            f"https://raw.githubusercontent.com/gradle/gradle/{tag}/gradle/wrapper/gradle-wrapper.jar",
        )
        live_pin = None
    else:
        live_pin = _live_wrapper_pin(gradle)
        url, sha, size = live_pin
        if target.is_file() and not target.is_symlink():
            if target.stat().st_size == size and _git_blob_sha1(target) == sha:
                _validate_wrapper(target)
                return target
        candidates = (url,)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    errors: list[str] = []
    for url in candidates:
        try:
            if temporary.exists():
                temporary.unlink()
            _download(url, temporary)
            if known and _sha256(temporary) != known:
                raise RuntimeError("wrapper SHA-256 mismatch")
            if live_pin:
                _, sha, size = live_pin
                if temporary.stat().st_size != size or _git_blob_sha1(temporary) != sha:
                    raise RuntimeError("wrapper Git object mismatch")
            _validate_wrapper(temporary)
            os.replace(temporary, target)
            return target
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Verified Gradle wrapper {gradle} unavailable: " + " | ".join(errors))


def _distribution_url(gradle: str) -> str:
    return f"https://services.gradle.org/distributions/gradle-{gradle}-bin.zip"


def _unobfuscated(version: str) -> bool:
    major = str(version).strip().split(".", 1)[0]
    return major.isdigit() and int(major) >= 26


def _loom_plugin(version: str) -> str:
    return "net.fabricmc.fabric-loom" + ("" if _unobfuscated(version) else "-remap")


def validate_scaffold_buildability(adapter: PlatformAdapter) -> None:
    """Validate the exact provider receipt before accepting it as buildable."""
    adapter.validate()
    if adapter.loader not in executable_loaders():
        raise UnsupportedTargetSpecificationError(f"No provider for loader={adapter.loader!r}")
    if adapter.loader != "fabric":
        raise UnsupportedTargetSpecificationError("Scaffold implementation is Fabric-only")
    if not re.fullmatch(r"[0-9][0-9A-Za-z_.-]*", adapter.gradle):
        raise UnsupportedTargetSpecificationError("Invalid Gradle version")
    if not re.fullmatch(r"[0-9a-f]{64}", adapter.gradle_sha256):
        raise UnsupportedTargetSpecificationError("Invalid Gradle SHA-256")
    try:
        java = int(adapter.java_version)
    except ValueError as exc:
        raise UnsupportedTargetSpecificationError("Invalid Java release") from exc
    if java <= 0 or not adapter.fabric_loader or not adapter.fabric_api or not adapter.fabric_loom:
        raise UnsupportedTargetSpecificationError("Incomplete Fabric build receipt")
    if adapter.mappings_kind not in {"mojang", "yarn"}:
        raise UnsupportedTargetSpecificationError("Unsupported mappings kind")


def is_target_supported(loader: str, minecraft_version: str) -> bool:
    try:
        validate_scaffold_buildability(adapter_for_target(minecraft_version, loader))
    except (ValueError, RuntimeError):
        return False
    return True


def get_verified_scaffold_template_for_adapter(adapter: PlatformAdapter) -> VerifiedScaffoldTemplate:
    validate_scaffold_buildability(adapter)
    java = int(adapter.java_version)
    unobfuscated = _unobfuscated(adapter.minecraft_version)
    plugin = _loom_plugin(adapter.minecraft_version)
    impl = "implementation" if unobfuscated else "modImplementation"
    if unobfuscated:
        mappings = ""
    elif adapter.mappings_kind == "mojang":
        mappings = "    mappings loom.officialMojangMappings()\n"
    else:
        mappings = f"    mappings 'net.fabricmc:yarn:{adapter.mappings_version}:v2'\n"
    build = f"""plugins {{
    id '{plugin}' version '{adapter.fabric_loom}'
    id 'maven-publish'
    id 'java'
}}

group = 'com.example'
version = '1.0.0'

repositories {{
    mavenCentral()
    maven {{ url 'https://maven.fabricmc.net/' }}
}}

dependencies {{
    minecraft 'com.mojang:minecraft:{adapter.minecraft_version}'
{mappings}    {impl} 'net.fabricmc:fabric-loader:{adapter.fabric_loader}'
    {impl} 'net.fabricmc.fabric-api:fabric-api:{adapter.fabric_api}'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = {java}
}}

java {{
    sourceCompatibility = JavaVersion.VERSION_{java}
    targetCompatibility = JavaVersion.VERSION_{java}
}}
"""
    settings = """pluginManagement {
    repositories {
        maven { url 'https://maven.fabricmc.net/' }
        mavenCentral()
        gradlePluginPortal()
    }
}
"""
    return VerifiedScaffoldTemplate(
        adapter.loader,
        adapter.minecraft_version,
        adapter.gradle,
        adapter.gradle_sha256,
        build,
        settings,
        adapter.adapter_id,
        java,
        plugin,
        _distribution_url(adapter.gradle),
    )


def get_verified_scaffold_template(
    loader: str = "fabric", minecraft_version: str = "1.21.1"
) -> VerifiedScaffoldTemplate:
    return get_verified_scaffold_template_for_adapter(adapter_for_target(minecraft_version, loader))


def _adapter_from_target_context(
    context: Mapping[str, Any] | PlatformAdapter,
) -> PlatformAdapter:
    if isinstance(context, PlatformAdapter):
        return context
    embedded = context.get("platform_adapter")
    if isinstance(embedded, PlatformAdapter):
        loader = str(context.get("loader") or embedded.loader).strip().casefold()
        version = str(context.get("minecraft_version") or embedded.minecraft_version).strip()
        if loader != embedded.loader or version != embedded.minecraft_version:
            raise UnsupportedTargetSpecificationError("Embedded adapter identity mismatch")
        return embedded
    return adapter_for_target(
        str(context.get("minecraft_version") or "1.21.1"),
        str(context.get("loader") or "fabric"),
    )


def apply_verified_scaffold_for_adapter(sandbox_path: Path, adapter: PlatformAdapter) -> None:
    template = get_verified_scaffold_template_for_adapter(adapter)
    wrapper = _ensure_wrapper(adapter)
    sandbox_path.mkdir(parents=True, exist_ok=True)
    build = sandbox_path / "build.gradle"
    if not build.exists() and not (sandbox_path / "build.gradle.kts").exists():
        build.write_text(template.build_gradle, encoding="utf-8")
    settings = sandbox_path / "settings.gradle"
    if not settings.exists() and not (sandbox_path / "settings.gradle.kts").exists():
        settings.write_text(template.settings_gradle, encoding="utf-8")
    gradle_props = sandbox_path / "gradle.properties"
    if not gradle_props.exists():
        gradle_props.write_text(
            "org.gradle.jvmargs=-Xmx2G\norg.gradle.parallel=true\n"
            "org.gradle.configuration-cache=false\n",
            encoding="utf-8",
        )
    wrapper_dir = sandbox_path / "gradle" / "wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    url = template.distribution_url.replace(":", "\\:", 1)
    (wrapper_dir / "gradle-wrapper.properties").write_text(
        "distributionBase=GRADLE_USER_HOME\ndistributionPath=wrapper/dists\n"
        f"distributionUrl={url}\ndistributionSha256Sum={template.distribution_sha256}\n"
        "networkTimeout=10000\nvalidateDistributionUrl=true\n"
        "zipStoreBase=GRADLE_USER_HOME\nzipStorePath=wrapper/dists\n",
        encoding="utf-8",
    )
    jar = wrapper_dir / "gradle-wrapper.jar"
    if jar.exists() or jar.is_symlink():
        jar.unlink()
    shutil.copy2(wrapper, jar)
    _validate_wrapper(jar)
    gradlew = sandbox_path / "gradlew"
    if not gradlew.exists():
        gradlew.write_text(
            "#!/bin/sh\n"
            'APP_HOME="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"\n'
            'CLASSPATH="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"\n'
            'JAVACMD="${JAVA_HOME:+$JAVA_HOME/bin/}java"\n'
            'exec "$JAVACMD" -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"\n',
            encoding="utf-8",
        )
        gradlew.chmod(0o755)
    bat = sandbox_path / "gradlew.bat"
    if not bat.exists():
        bat.write_text(
            "@echo off\r\nset DIRNAME=%~dp0\r\nset CLASSPATH=%DIRNAME%gradle\\wrapper\\gradle-wrapper.jar\r\n"
            "set JAVACMD=java.exe\r\nif defined JAVA_HOME set JAVACMD=%JAVA_HOME%\\bin\\java.exe\r\n"
            '"%JAVACMD%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*\r\n',
            encoding="utf-8",
        )


def apply_verified_scaffold(
    sandbox_path: Path, target_context: Mapping[str, Any] | PlatformAdapter
) -> None:
    apply_verified_scaffold_for_adapter(sandbox_path, _adapter_from_target_context(target_context))


__all__ = [
    "GRADLE_CACHE_RELEASE_TAG",
    "GRADLE_DISTRIBUTION_SHA256S",
    "GRADLE_WRAPPER_SHA256S",
    "UnsupportedTargetSpecificationError",
    "VerifiedScaffoldTemplate",
    "apply_verified_scaffold",
    "apply_verified_scaffold_for_adapter",
    "get_verified_scaffold_template",
    "get_verified_scaffold_template_for_adapter",
    "is_target_supported",
    "validate_scaffold_buildability",
]
