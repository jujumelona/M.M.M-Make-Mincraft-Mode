from __future__ import annotations

"""Verified Gradle scaffold registry for isolated reuse-proof sandboxes.

The scaffold never fabricates a wrapper JAR. Each supported Gradle version is bound to
its official distribution and wrapper SHA-256. M.M.M mirrors those official artifacts to
an append-only GitHub Release cache and validates the wrapper again before materializing it.
"""

import hashlib
import os
import shutil
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRADLE_CACHE_RELEASE_TAG = "gradle-runtime-cache-v1-immutable"
GRADLE_CACHE_RELEASE_BASE = (
    "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode/releases/download/"
    + GRADLE_CACHE_RELEASE_TAG
)

GRADLE_DISTRIBUTION_SHA256S: dict[str, str] = {
    "7.6": "7ba68c54029790ab444b39d7e293d3236b2632631fb5f2e012bb28b4ff669e4b",
    "8.1": "a62c5f99585dd9e1f95dab7b9415a0e698fa9dd1e6c38537faa81ac078f4d23e",
    "8.5": "9d926787066a081739e8200858338b4a69e837c3a821a33aca9db09dd4a41026",
    "8.8": "a4b4158601f8636cdeeab09bd76afb640030bb5b144aafe261a5e8af027dc612",
    "8.10.2": "31c55713e40233a8303827ceb42ca48a47267a0ad4bab9177123121e71524c26",
}

GRADLE_WRAPPER_SHA256S: dict[str, str] = {
    "7.6": "c5a643cf80162e665cc228f7b16f343fef868e47d3a4836f62e18b7e17ac018a",
    "8.1": "ed2c26eba7cfb93cc2b7785d05e534f07b5b48b5e7fc941921cd098628abca58",
    "8.5": "d3b261c2820e9e3d8d639ed084900f11f4a86050a8f83342ade7b6bc9b0d2bdd",
    "8.8": "cb0da6751c2b753a16ac168bb354870ebb1e162e9083f116729cec9c781156b8",
    "8.10.2": "2db75c40782f5e8ba1fc278a5574bab070adccb2d21ca5a6e5ed840888448046",
}

GRADLE_SOURCE_TAGS: dict[str, str] = {
    "7.6": "v7.6.0",
    "8.1": "v8.1.0",
    "8.5": "v8.5.0",
    "8.8": "v8.8.0",
    "8.10.2": "v8.10.2",
}


@dataclass(frozen=True)
class VerifiedScaffoldTemplate:
    loader: str
    minecraft_version: str
    gradle_version: str
    distribution_sha256: str
    build_gradle: str
    settings_gradle: str


SUPPORTED_TARGET_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("fabric", "1.21.4"): {"gradle_version": "8.10.2", "loom_version": "1.9-SNAPSHOT", "loader_version": "0.16.9", "fabric_api": "0.110.0+1.21.4", "mappings": "net.fabricmc:yarn:1.21.4+build.1:v2", "java_release": 21},
    ("fabric", "1.21.3"): {"gradle_version": "8.10.2", "loom_version": "1.8-SNAPSHOT", "loader_version": "0.16.7", "fabric_api": "0.108.0+1.21.3", "mappings": "net.fabricmc:yarn:1.21.3+build.1:v2", "java_release": 21},
    ("fabric", "1.21.2"): {"gradle_version": "8.10.2", "loom_version": "1.8-SNAPSHOT", "loader_version": "0.16.7", "fabric_api": "0.107.0+1.21.2", "mappings": "net.fabricmc:yarn:1.21.2+build.1:v2", "java_release": 21},
    ("fabric", "1.21.1"): {"gradle_version": "8.10.2", "loom_version": "1.7-SNAPSHOT", "loader_version": "0.16.5", "fabric_api": "0.104.0+1.21.1", "mappings": "net.fabricmc:yarn:1.21.1+build.1:v2", "java_release": 21},
    ("fabric", "1.21.0"): {"gradle_version": "8.10.2", "loom_version": "1.7-SNAPSHOT", "loader_version": "0.16.0", "fabric_api": "0.100.0+1.21", "mappings": "net.fabricmc:yarn:1.21+build.9:v2", "java_release": 21},
    ("fabric", "1.20.6"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.15.11", "fabric_api": "0.99.1+1.20.6", "mappings": "net.fabricmc:yarn:1.20.6+build.2:v2", "java_release": 21},
    ("fabric", "1.20.5"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.15.11", "fabric_api": "0.97.8+1.20.5", "mappings": "net.fabricmc:yarn:1.20.5+build.1:v2", "java_release": 21},
    ("fabric", "1.20.4"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.15.7", "fabric_api": "0.96.11+1.20.4", "mappings": "net.fabricmc:yarn:1.20.4+build.3:v2", "java_release": 17},
    ("fabric", "1.20.2"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.15.0", "fabric_api": "0.91.6+1.20.2", "mappings": "net.fabricmc:yarn:1.20.2+build.4:v2", "java_release": 17},
    ("fabric", "1.20.1"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.15.11", "fabric_api": "0.92.2+1.20.1", "mappings": "net.fabricmc:yarn:1.20.1+build.10:v2", "java_release": 17},
    ("fabric", "1.20.0"): {"gradle_version": "8.8", "loom_version": "1.6-SNAPSHOT", "loader_version": "0.14.21", "fabric_api": "0.83.0+1.20", "mappings": "net.fabricmc:yarn:1.20+build.1:v2", "java_release": 17},
    ("fabric", "1.19.4"): {"gradle_version": "8.5", "loom_version": "1.5-SNAPSHOT", "loader_version": "0.14.21", "fabric_api": "0.87.0+1.19.4", "mappings": "net.fabricmc:yarn:1.19.4+build.2:v2", "java_release": 17},
    ("fabric", "1.19.2"): {"gradle_version": "8.5", "loom_version": "1.5-SNAPSHOT", "loader_version": "0.14.21", "fabric_api": "0.76.0+1.19.2", "mappings": "net.fabricmc:yarn:1.19.2+build.28:v2", "java_release": 17},
    ("fabric", "1.18.2"): {"gradle_version": "8.1", "loom_version": "1.4-SNAPSHOT", "loader_version": "0.14.21", "fabric_api": "0.76.0+1.18.2", "mappings": "net.fabricmc:yarn:1.18.2+build.4:v2", "java_release": 17},
    ("fabric", "1.16.5"): {"gradle_version": "7.6", "loom_version": "1.0-SNAPSHOT", "loader_version": "0.14.21", "fabric_api": "0.42.0+1.16", "mappings": "net.fabricmc:yarn:1.16.5+build.10:v2", "java_release": 8},
    ("neoforge", "1.21.4"): {"gradle_version": "8.10.2", "moddev_version": "2.0.80", "neoforge_version": "21.4.0", "java_release": 21},
    ("neoforge", "1.21.3"): {"gradle_version": "8.10.2", "moddev_version": "2.0.79", "neoforge_version": "21.3.0", "java_release": 21},
    ("neoforge", "1.21.1"): {"gradle_version": "8.10.2", "moddev_version": "2.0.78", "neoforge_version": "1.21.1-21.1.0", "java_release": 21},
    ("neoforge", "1.21.0"): {"gradle_version": "8.10.2", "moddev_version": "2.0.78", "neoforge_version": "21.0.167", "java_release": 21},
    ("neoforge", "1.20.6"): {"gradle_version": "8.8", "moddev_version": "2.0.74", "neoforge_version": "20.6.119", "java_release": 21},
    ("neoforge", "1.20.4"): {"gradle_version": "8.8", "moddev_version": "2.0.70", "neoforge_version": "20.4.237", "java_release": 17},
    ("forge", "1.21.1"): {"gradle_version": "8.8", "forgegradle_version": "6.0.29", "forge_version": "51.0.8", "java_release": 21},
    ("forge", "1.21.0"): {"gradle_version": "8.8", "forgegradle_version": "6.0.29", "forge_version": "51.0.0", "java_release": 21},
    ("forge", "1.20.4"): {"gradle_version": "8.8", "forgegradle_version": "6.0.29", "forge_version": "49.0.38", "java_release": 17},
    ("forge", "1.20.2"): {"gradle_version": "8.8", "forgegradle_version": "6.0.29", "forge_version": "48.1.0", "java_release": 17},
    ("forge", "1.20.1"): {"gradle_version": "8.8", "forgegradle_version": "6.0.29", "forge_version": "47.3.0", "java_release": 17},
    ("forge", "1.19.4"): {"gradle_version": "8.5", "forgegradle_version": "6.0.18", "forge_version": "45.2.0", "java_release": 17},
    ("forge", "1.19.2"): {"gradle_version": "8.5", "forgegradle_version": "6.0.16", "forge_version": "43.3.0", "java_release": 17},
    ("forge", "1.18.2"): {"gradle_version": "8.1", "forgegradle_version": "5.1.74", "forge_version": "40.2.0", "java_release": 17},
    ("forge", "1.16.5"): {"gradle_version": "7.6", "forgegradle_version": "5.1.69", "forge_version": "36.2.39", "java_release": 8},
}


class UnsupportedTargetSpecificationError(ValueError):
    """Raised when a loader/Minecraft version is outside the verified matrix."""


def is_target_supported(loader: str, minecraft_version: str) -> bool:
    return (loader.lower().strip(), minecraft_version.strip()) in SUPPORTED_TARGET_SPECS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrapper_cache_root() -> Path:
    override = os.environ.get("MMM_GRADLE_WRAPPER_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".cache" / "mmm" / "gradle-wrapper").resolve()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "M.M.M-verified-gradle-wrapper/1"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def _ensure_verified_wrapper_jar(gradle_version: str) -> Path:
    expected = GRADLE_WRAPPER_SHA256S.get(gradle_version)
    source_tag = GRADLE_SOURCE_TAGS.get(gradle_version)
    if not expected or not source_tag:
        raise UnsupportedTargetSpecificationError(
            f"No verified Gradle wrapper checksum for {gradle_version}"
        )

    target = _wrapper_cache_root() / gradle_version / "gradle-wrapper.jar"
    if target.is_file() and not target.is_symlink() and _sha256(target) == expected:
        return target

    if target.exists() or target.is_symlink():
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jar.tmp")
    urls = (
        f"{GRADLE_CACHE_RELEASE_BASE}/gradle-{gradle_version}-wrapper.jar",
        "https://raw.githubusercontent.com/gradle/gradle/"
        f"{source_tag}/gradle/wrapper/gradle-wrapper.jar",
    )
    errors: list[str] = []
    for url in urls:
        try:
            if temporary.exists():
                temporary.unlink()
            _download(url, temporary)
            actual = _sha256(temporary)
            if actual != expected:
                raise RuntimeError(
                    f"Gradle wrapper checksum mismatch for {gradle_version}: "
                    f"expected {expected}, found {actual}"
                )
            os.replace(temporary, target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if temporary.exists():
                temporary.unlink()
    raise RuntimeError(
        f"Verified Gradle wrapper {gradle_version} is unavailable from GitHub cache/source: "
        + " | ".join(errors)
    )


def _distribution_url(gradle_version: str) -> str:
    return f"{GRADLE_CACHE_RELEASE_BASE}/gradle-{gradle_version}-bin.zip"


def get_verified_scaffold_template(
    loader: str = "fabric",
    minecraft_version: str = "1.21.1",
) -> VerifiedScaffoldTemplate:
    norm_loader = loader.lower().strip()
    norm_mc = minecraft_version.strip()
    key = (norm_loader, norm_mc)
    if key not in SUPPORTED_TARGET_SPECS:
        raise UnsupportedTargetSpecificationError(
            f"Target ({norm_loader}@{norm_mc}) is not in SUPPORTED_TARGET_SPECS: "
            f"{sorted(SUPPORTED_TARGET_SPECS.keys())}"
        )
    spec = SUPPORTED_TARGET_SPECS[key]
    gradle_ver = str(spec["gradle_version"])
    distribution_sha256 = GRADLE_DISTRIBUTION_SHA256S.get(gradle_ver)
    if not distribution_sha256:
        raise UnsupportedTargetSpecificationError(
            f"No verified Gradle distribution checksum for {gradle_ver}"
        )
    java_rel = int(spec["java_release"])

    if norm_loader == "neoforge":
        bg = f"""plugins {{
    id 'net.neoforged.moddev' version '{spec['moddev_version']}'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

neoForge {{
    version = '{spec['neoforge_version']}'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = {java_rel}
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    elif norm_loader == "forge":
        bg = f"""plugins {{
    id 'net.minecraftforge.gradle' version '{spec['forgegradle_version']}'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

minecraft {{
    mappings channel: 'official', version: '{norm_mc}'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = {java_rel}
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    else:
        bg = f"""plugins {{
    id 'fabric-loom' version '{spec['loom_version']}'
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
    minecraft 'com.mojang:minecraft:{norm_mc}'
    mappings '{spec['mappings']}'
    modImplementation 'net.fabricmc:fabric-loader:{spec['loader_version']}'
    modImplementation 'net.fabricmc.fabric-api:fabric-api:{spec['fabric_api']}'
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
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
        loader=norm_loader,
        minecraft_version=norm_mc,
        gradle_version=gradle_ver,
        distribution_sha256=distribution_sha256,
        build_gradle=bg,
        settings_gradle=settings,
    )


def apply_verified_scaffold(
    sandbox_path: Path,
    target_context: Mapping[str, Any],
) -> None:
    """Materialize a checksum-bound Gradle scaffold into ``sandbox_path``."""

    loader = str(target_context.get("loader") or "fabric")
    mc_ver = str(target_context.get("minecraft_version") or "1.21.1")
    template = get_verified_scaffold_template(loader, mc_ver)
    wrapper_source = _ensure_verified_wrapper_jar(template.gradle_version)
    expected_wrapper_sha = GRADLE_WRAPPER_SHA256S[template.gradle_version]

    sandbox_path.mkdir(parents=True, exist_ok=True)

    bg = sandbox_path / "build.gradle"
    if not bg.exists() and not (sandbox_path / "build.gradle.kts").exists():
        bg.write_text(template.build_gradle, encoding="utf-8")

    settings = sandbox_path / "settings.gradle"
    if not settings.exists() and not (sandbox_path / "settings.gradle.kts").exists():
        settings.write_text(template.settings_gradle, encoding="utf-8")

    gradle_props = sandbox_path / "gradle.properties"
    if not gradle_props.exists():
        gradle_props.write_text("org.gradle.jvmargs=-Xmx2G\n", encoding="utf-8")

    wrapper_dir = sandbox_path / "gradle" / "wrapper"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    props = wrapper_dir / "gradle-wrapper.properties"
    distribution_url = _distribution_url(template.gradle_version).replace(":", "\\:", 1)
    props.write_text(
        "distributionBase=GRADLE_USER_HOME\n"
        "distributionPath=wrapper/dists\n"
        f"distributionUrl={distribution_url}\n"
        f"distributionSha256Sum={template.distribution_sha256}\n"
        "zipStoreBase=GRADLE_USER_HOME\n"
        "zipStorePath=wrapper/dists\n",
        encoding="utf-8",
    )

    jar = wrapper_dir / "gradle-wrapper.jar"
    if (
        not jar.is_file()
        or jar.is_symlink()
        or _sha256(jar) != expected_wrapper_sha
    ):
        if jar.exists() or jar.is_symlink():
            jar.unlink()
        shutil.copy2(wrapper_source, jar)
    if _sha256(jar) != expected_wrapper_sha:
        raise RuntimeError("Materialized Gradle wrapper checksum verification failed")

    gradlew_sh = sandbox_path / "gradlew"
    if not gradlew_sh.exists():
        gradlew_sh.write_text(
            "#!/bin/sh\n"
            "APP_HOME=\"$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd -P)\"\n"
            "CLASSPATH=\"$APP_HOME/gradle/wrapper/gradle-wrapper.jar\"\n"
            "if [ -n \"${JAVA_HOME:-}\" ] && [ -x \"$JAVA_HOME/bin/java\" ]; then\n"
            "    JAVACMD=\"$JAVA_HOME/bin/java\"\n"
            "else\n"
            "    JAVACMD=java\n"
            "fi\n"
            "exec \"$JAVACMD\" -classpath \"$CLASSPATH\" org.gradle.wrapper.GradleWrapperMain \"$@\"\n",
            encoding="utf-8",
        )
        gradlew_sh.chmod(0o755)

    gradlew_bat = sandbox_path / "gradlew.bat"
    if not gradlew_bat.exists():
        gradlew_bat.write_text(
            "@if \"%DEBUG%\"==\"\" @echo off\r\n"
            "setlocal\r\n"
            "set DIRNAME=%~dp0\r\n"
            "if \"%DIRNAME%\"==\"\" set DIRNAME=.\r\n"
            "set APP_HOME=%DIRNAME%\r\n"
            "set CLASSPATH=%APP_HOME%\\gradle\\wrapper\\gradle-wrapper.jar\r\n"
            "set JAVACMD=java.exe\r\n"
            "if defined JAVA_HOME if exist \"%JAVA_HOME%\\bin\\java.exe\" set JAVACMD=%JAVA_HOME%\\bin\\java.exe\r\n"
            "\"%JAVACMD%\" -classpath \"%CLASSPATH%\" org.gradle.wrapper.GradleWrapperMain %*\r\n",
            encoding="utf-8",
        )


__all__ = [
    "GRADLE_CACHE_RELEASE_TAG",
    "GRADLE_DISTRIBUTION_SHA256S",
    "GRADLE_WRAPPER_SHA256S",
    "SUPPORTED_TARGET_SPECS",
    "UnsupportedTargetSpecificationError",
    "VerifiedScaffoldTemplate",
    "apply_verified_scaffold",
    "get_verified_scaffold_template",
    "is_target_supported",
]
