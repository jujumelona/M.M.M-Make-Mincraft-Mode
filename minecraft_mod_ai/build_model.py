from __future__ import annotations

"""Authoritative Build Model & Deterministic Gradle Renderer.

Unifies build target specifications, dependency coordinates, Maven repositories,
plugins, and source sets into a single authoritative object model, replacing
string regex injections with structured Gradle generation.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResolvedRepository:
    url: str
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url, "name": self.name}


@dataclass(frozen=True)
class ResolvedDependency:
    configuration: str  # e.g., "modImplementation", "implementation", "compileOnly"
    coordinate: str     # e.g., "net.fabricmc.fabric-api:fabric-api:0.100.0+1.21.1"
    sha256: str = ""
    source_requirement_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "coordinate": self.coordinate,
            "sha256": self.sha256,
            "source_requirement_ids": list(self.source_requirement_ids),
        }


@dataclass(frozen=True)
class BuildTargetSpec:
    loader: str
    minecraft_version: str
    java_version: int = 21
    gradle_version: str = "8.10.2"
    plugin_versions: Mapping[str, str] = field(default_factory=dict)
    mappings: str = "yarn"
    loader_version: str = ""
    platform_api_version: str = ""
    platform_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "loader": self.loader,
            "minecraft_version": self.minecraft_version,
            "java_version": self.java_version,
            "gradle_version": self.gradle_version,
            "plugin_versions": dict(self.plugin_versions),
            "mappings": self.mappings,
            "loader_version": self.loader_version,
            "platform_api_version": self.platform_api_version,
            "platform_version": self.platform_version,
        }


@dataclass
class BuildModel:
    target: BuildTargetSpec
    repositories: list[ResolvedRepository] = field(default_factory=list)
    dependencies: list[ResolvedDependency] = field(default_factory=list)
    plugins: list[str] = field(default_factory=list)
    source_sets: list[str] = field(default_factory=lambda: ["main", "client"])

    @classmethod
    def for_target_context(cls, target_context: Mapping[str, Any]) -> BuildModel:
        """Create a build model from the same verified target registry as scaffolding."""
        from .verified_scaffold_registry import (
            SUPPORTED_TARGET_SPECS,
            UnsupportedTargetSpecificationError,
        )

        loader = str(target_context.get("loader") or "fabric").casefold().strip()
        minecraft_version = str(
            target_context.get("minecraft_version") or "1.21.1"
        ).strip()
        key = (loader, minecraft_version)
        if key not in SUPPORTED_TARGET_SPECS:
            raise UnsupportedTargetSpecificationError(
                f"Target ({loader}@{minecraft_version}) has no verified build model"
            )
        verified = SUPPORTED_TARGET_SPECS[key]

        plugin_versions: dict[str, str] = {}
        if loader == "fabric":
            plugin_versions["fabric-loom"] = str(verified["loom_version"])
        elif loader == "neoforge":
            plugin_versions["neoforge-moddev"] = str(verified["moddev_version"])
        elif loader == "forge":
            plugin_versions["forgegradle"] = str(verified["forgegradle_version"])

        return cls(
            target=BuildTargetSpec(
                loader=loader,
                minecraft_version=minecraft_version,
                java_version=int(verified["java_release"]),
                gradle_version=str(verified["gradle_version"]),
                plugin_versions=plugin_versions,
                mappings=str(verified.get("mappings") or "official"),
                loader_version=str(verified.get("loader_version") or ""),
                platform_api_version=str(verified.get("fabric_api") or ""),
                platform_version=str(
                    verified.get("neoforge_version")
                    or verified.get("forge_version")
                    or ""
                ),
            )
        )

    def add_repository(self, url: str, name: str = "") -> None:
        if not any(r.url == url for r in self.repositories):
            self.repositories.append(ResolvedRepository(url=url, name=name))

    def add_dependency(
        self,
        coordinate: str,
        configuration: str = "modImplementation",
        *,
        sha256: str = "",
        requirement_ids: Sequence[str] = (),
    ) -> None:
        if not any(d.coordinate == coordinate for d in self.dependencies):
            self.dependencies.append(
                ResolvedDependency(
                    configuration=configuration,
                    coordinate=coordinate,
                    sha256=sha256,
                    source_requirement_ids=tuple(requirement_ids),
                )
            )

    def render_gradle(self, *, modid: str = "generated_mod", version: str = "1.0.0") -> str:
        """Render standard build.gradle from this authoritative build model."""
        lines: list[str] = []

        # Plugins block
        lines.append("plugins {")
        if self.target.loader == "fabric":
            loom_ver = self.target.plugin_versions.get("fabric-loom", "1.9-SNAPSHOT")
            lines.append(f'    id "fabric-loom" version "{loom_ver}"')
        elif self.target.loader == "neoforge":
            moddev_ver = self.target.plugin_versions.get("neoforge-moddev", "2.0.78")
            lines.append(f'    id "net.neoforged.moddev" version "{moddev_ver}"')
        elif self.target.loader == "forge":
            forgegradle_ver = self.target.plugin_versions.get("forgegradle", "6.0.29")
            lines.append(f'    id "net.minecraftforge.gradle" version "{forgegradle_ver}"')
        lines.append('    id "maven-publish"')
        lines.append("}")
        lines.append("")

        lines.append(f'version = "{version}"')
        lines.append(f'group = "ai.minecraft.{modid}"')
        lines.append(f'base.archivesName = "{modid}"')
        lines.append("")

        # Repositories block
        lines.append("repositories {")
        lines.append("    mavenCentral()")
        if self.target.loader == "fabric":
            lines.append('    maven { url = "https://maven.fabricmc.net/" }')
        elif self.target.loader == "neoforge":
            lines.append('    maven { url = "https://maven.neoforged.net/releases/" }')
        elif self.target.loader == "forge":
            lines.append('    maven { url = "https://maven.minecraftforge.net/" }')
        for repo in self.repositories:
            lines.append(f'    maven {{ url = "{repo.url}" }}')
        lines.append("}")
        lines.append("")

        # Dependencies block
        lines.append("dependencies {")
        if self.target.loader == "fabric":
            mappings = self.target.mappings
            if ":" not in mappings:
                mappings = f"net.fabricmc:yarn:{self.target.minecraft_version}+build.1:v2"
            loader_version = self.target.loader_version or "0.16.9"
            api_version = (
                self.target.platform_api_version
                or f"0.108.0+{self.target.minecraft_version}"
            )
            lines.append(
                f'    minecraft "com.mojang:minecraft:{self.target.minecraft_version}"'
            )
            lines.append(f'    mappings "{mappings}"')
            lines.append(
                f'    modImplementation "net.fabricmc:fabric-loader:{loader_version}"'
            )
            lines.append(
                '    modImplementation '
                f'"net.fabricmc.fabric-api:fabric-api:{api_version}"'
            )
        elif self.target.loader == "forge" and self.target.platform_version:
            lines.append(
                '    minecraft '
                f'"net.minecraftforge:forge:{self.target.minecraft_version}-'
                f'{self.target.platform_version}"'
            )

        for dep in self.dependencies:
            lines.append(f'    {dep.configuration} "{dep.coordinate}"')

        lines.append('    testImplementation "org.junit.jupiter:junit-jupiter:5.10.2"')
        lines.append("}")
        lines.append("")

        if self.target.loader == "neoforge" and self.target.platform_version:
            lines.append("neoForge {")
            lines.append(f'    version = "{self.target.platform_version}"')
            lines.append("}")
            lines.append("")
        elif self.target.loader == "forge":
            lines.append("minecraft {")
            lines.append(
                f'    mappings channel: "official", version: "{self.target.minecraft_version}"'
            )
            lines.append("}")
            lines.append("")

        # Java toolchain block
        lines.append("java {")
        lines.append(f"    sourceCompatibility = JavaVersion.VERSION_{self.target.java_version}")
        lines.append(f"    targetCompatibility = JavaVersion.VERSION_{self.target.java_version}")
        lines.append("}")
        lines.append("")

        lines.append("tasks.withType(JavaCompile).configureEach {")
        lines.append('    options.encoding = "UTF-8"')
        lines.append(f"    options.release = {self.target.java_version}")
        lines.append("}")
        lines.append("")

        lines.append("test {")
        lines.append("    useJUnitPlatform()")
        lines.append("}")

        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmm/build-model-v1",
            "target": self.target.to_dict(),
            "repositories": [r.to_dict() for r in self.repositories],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "plugins": list(self.plugins),
            "source_sets": list(self.source_sets),
        }
