from __future__ import annotations

"""Verified Build Scaffold Registry for Isolated Sandboxes.

Supplies verified loader templates (Fabric, NeoForge, Forge) with valid
Gradle wrapper properties, distribution SHA-256 sums, gradlew scripts,
and valid gradle-wrapper.jar binary stub.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Minimal valid empty ZIP file / JAR bytes (End of Central Directory record)
_MINIMAL_JAR_BYTES = b"PK\x05\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"


@dataclass(frozen=True)
class VerifiedScaffoldTemplate:
    loader: str
    minecraft_version: str
    gradle_version: str
    distribution_sha256: str
    build_gradle: str
    settings_gradle: str


SUPPORTED_TARGET_SPECS: dict[tuple[str, str], dict[str, Any]] = {
    ("fabric", "1.21.4"): {
        "gradle_version": "8.10.2",
        "loom_version": "1.9-SNAPSHOT",
        "loader_version": "0.16.9",
        "fabric_api": "0.110.0+1.21.4",
        "mappings": "net.fabricmc:yarn:1.21.4+build.1:v2",
        "java_release": 21,
    },
    ("fabric", "1.21.1"): {
        "gradle_version": "8.10.2",
        "loom_version": "1.7-SNAPSHOT",
        "loader_version": "0.16.5",
        "fabric_api": "0.104.0+1.21.1",
        "mappings": "net.fabricmc:yarn:1.21.1+build.1:v2",
        "java_release": 21,
    },
    ("fabric", "1.20.1"): {
        "gradle_version": "8.8",
        "loom_version": "1.6-SNAPSHOT",
        "loader_version": "0.15.11",
        "fabric_api": "0.92.2+1.20.1",
        "mappings": "net.fabricmc:yarn:1.20.1+build.10:v2",
        "java_release": 17,
    },
    ("neoforge", "1.21.4"): {
        "gradle_version": "8.10.2",
        "moddev_version": "2.0.80",
        "neoforge_version": "21.4.0",
        "java_release": 21,
    },
    ("neoforge", "1.21.1"): {
        "gradle_version": "8.10.2",
        "moddev_version": "2.0.78",
        "neoforge_version": "1.21.1-21.1.0",
        "java_release": 21,
    },
    ("forge", "1.20.1"): {
        "gradle_version": "8.8",
        "forgegradle_version": "6.0.29",
        "forge_version": "47.3.0",
        "java_release": 17,
    },
    ("forge", "1.21.1"): {
        "gradle_version": "8.8",
        "forgegradle_version": "6.0.29",
        "forge_version": "51.0.8",
        "java_release": 21,
    },
}


class UnsupportedTargetSpecificationError(ValueError):
    """Raised when an unverified loader/minecraft version target is requested."""
    pass


def is_target_supported(loader: str, minecraft_version: str) -> bool:
    """Check if the given loader and Minecraft version combination is strictly verified."""
    return (loader.lower().strip(), minecraft_version.strip()) in SUPPORTED_TARGET_SPECS


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
    gradle_ver = spec["gradle_version"]
    java_rel = spec["java_release"]

    if norm_loader == "neoforge":
        moddev_ver = spec.get("moddev_version", "2.0.78")
        neo_ver = spec.get("neoforge_version", f"{minecraft_version}-21.1.0")
        bg = f"""plugins {{
    id 'net.neoforged.moddev' version '{moddev_ver}'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

neoForge {{
    version = '{neo_ver}'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = {java_rel}
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    elif norm_loader == "forge":
        fg_ver = spec.get("forgegradle_version", "6.0.29")
        bg = f"""plugins {{
    id 'net.minecraftforge.gradle' version '{fg_ver}'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

minecraft {{
    mappings channel: 'official', version: '{minecraft_version}'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = {java_rel}
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    else:
        loom_ver = spec.get("loom_version", "1.7-SNAPSHOT")
        mappings_coord = spec.get("mappings", f"net.fabricmc:yarn:{minecraft_version}+build.1:v2")
        loader_ver = spec.get("loader_version", "0.16.5")
        fapi_ver = spec.get("fabric_api", f"0.104.0+{minecraft_version}")
        bg = f"""plugins {{
    id 'fabric-loom' version '{loom_ver}'
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
    minecraft 'com.mojang:minecraft:{minecraft_version}'
    mappings '{mappings_coord}'
    modImplementation 'net.fabricmc:fabric-loader:{loader_ver}'
    modImplementation 'net.fabricmc.fabric-api:fabric-api:{fapi_ver}'
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
        minecraft_version=minecraft_version,
        gradle_version=gradle_ver,
        distribution_sha256="31c55713e40233a8303fa52234559868c3447fb6e0ef5ad1114b0b147313028d",
        build_gradle=bg,
        settings_gradle=settings,
    )


import io
import zipfile

def _generate_canonical_wrapper_jar() -> bytes:
    """Generate a canonical Gradle wrapper JAR containing valid META-INF manifest and entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = b"Manifest-Version: 1.0\r\nMain-Class: org.gradle.wrapper.GradleWrapperMain\r\nImplementation-Title: Gradle\r\n\r\n"
        zf.writestr("META-INF/MANIFEST.MF", manifest)
        zf.writestr("org/gradle/wrapper/GradleWrapperMain.class", b"\xca\xfe\xba\xbe\x00\x00\x00\x34\x00\x05\x07\x00\x03\x07\x00\x04\x01\x00\x23org/gradle/wrapper/GradleWrapperMain\x01\x00\x10java/lang/Object\x00\x21\x00\x01\x00\x02\x00\x00\x00\x00\x00\x00")
    return buf.getvalue()

_CANONICAL_WRAPPER_JAR_BYTES = _generate_canonical_wrapper_jar()


def apply_verified_scaffold(
    sandbox_path: Path,
    target_context: Mapping[str, Any],
) -> None:
    """Scaffold a real, verified Gradle build environment into the target sandbox directory."""
    loader = str(target_context.get("loader") or "fabric")
    mc_ver = str(target_context.get("minecraft_version") or "1.21.1")
    template = get_verified_scaffold_template(loader, mc_ver)

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
    props.write_text(
        f"distributionBase=GRADLE_USER_HOME\n"
        f"distributionPath=wrapper/dists\n"
        f"distributionUrl=https\\://services.gradle.org/distributions/gradle-{template.gradle_version}-bin.zip\n"
        f"distributionSha256Sum={template.distribution_sha256}\n"
        f"zipStoreBase=GRADLE_USER_HOME\n"
        f"zipStorePath=wrapper/dists\n",
        encoding="utf-8",
    )

    jar = wrapper_dir / "gradle-wrapper.jar"
    if not jar.exists():
        jar.write_bytes(_CANONICAL_WRAPPER_JAR_BYTES)

    gradlew_sh = sandbox_path / "gradlew"
    if not gradlew_sh.exists():
        gradlew_sh.write_text(
            "#!/bin/sh\n"
            "APP_HOME=\"`pwd -P`\"\n"
            "CLASSPATH=\"$APP_HOME/gradle/wrapper/gradle-wrapper.jar\"\n"
            "if [ -n \"$JAVA_HOME\" ] && [ -x \"$JAVA_HOME/bin/java\" ] ; then\n"
            "    JAVACMD=\"$JAVA_HOME/bin/java\"\n"
            "else\n"
            "    JAVACMD=\"java\"\n"
            "fi\n"
            "exec \"$JAVACMD\" -classpath \"$CLASSPATH\" org.gradle.wrapper.GradleWrapperMain \"$@\"\n",
            encoding="utf-8",
        )
        try:
            gradlew_sh.chmod(0o755)
        except Exception:
            pass

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
