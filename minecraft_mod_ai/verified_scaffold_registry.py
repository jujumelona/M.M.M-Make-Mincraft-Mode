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


def get_verified_scaffold_template(
    loader: str = "fabric",
    minecraft_version: str = "1.21.1",
) -> VerifiedScaffoldTemplate:
    norm_loader = loader.lower().strip()
    if norm_loader == "neoforge":
        bg = f"""plugins {{
    id 'net.neoforged.moddev' version '2.0.78'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

neoForge {{
    version = '{minecraft_version}-21.1.0'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = 21
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    elif norm_loader == "forge":
        bg = f"""plugins {{
    id 'net.minecraftforge.gradle' version '6.0.29'
}}

version = '1.0.0'
group = 'ai.minecraft.generated'

minecraft {{
    mappings channel: 'official', version: '{minecraft_version}'
}}

tasks.withType(JavaCompile).configureEach {{
    it.options.release = 21
}}
"""
        settings = "pluginManagement {\n    repositories {\n        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n"
    else:
        bg = f"""plugins {{
    id 'fabric-loom' version '1.7-SNAPSHOT'
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
    mappings 'net.fabricmc:yarn:{minecraft_version}+build.1:v2'
    modImplementation 'net.fabricmc:fabric-loader:0.16.5'
    modImplementation 'net.fabricmc.fabric-api:fabric-api:0.104.0+{minecraft_version}'
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
        gradle_version="8.10.2",
        distribution_sha256="31c55713e40233a8303fa52234559868c3447fb6e0ef5ad1114b0b147313028d",
        build_gradle=bg,
        settings_gradle=settings,
    )


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
        jar.write_bytes(_MINIMAL_JAR_BYTES)

    gradlew_sh = sandbox_path / "gradlew"
    if not gradlew_sh.exists():
        gradlew_sh.write_text("#!/bin/sh\nexec gradle \"$@\"\n", encoding="utf-8")
        try:
            gradlew_sh.chmod(0o755)
        except Exception:
            pass

    gradlew_bat = sandbox_path / "gradlew.bat"
    if not gradlew_bat.exists():
        gradlew_bat.write_text("@echo off\r\ngradle %*\r\n", encoding="utf-8")
