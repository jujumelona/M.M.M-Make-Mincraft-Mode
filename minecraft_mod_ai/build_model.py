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
    mappings_kind: str = "yarn"
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
            "mappings_kind": self.mappings_kind,
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
        """Create a build model from the exact provider receipt used by scaffolding."""
        from .verified_scaffold_registry import (
            _adapter_from_target_context,
            validate_scaffold_buildability,
        )

        adapter = _adapter_from_target_context(target_context)
        validate_scaffold_buildability(adapter)

        plugin_versions = {"fabric-loom": str(adapter.fabric_loom)}
        return cls(
            target=BuildTargetSpec(
                loader=adapter.loader,
                minecraft_version=adapter.minecraft_version,
                java_version=int(adapter.java_version),
                gradle_version=adapter.gradle,
                plugin_versions=plugin_versions,
                mappings=adapter.mappings_version,
                mappings_kind=adapter.mappings_kind,
                loader_version=adapter.fabric_loader,
                platform_api_version=adapter.fabric_api,
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

    @staticmethod
    def _unobfuscated_minecraft(version: str) -> bool:
        major = str(version).strip().split(".", 1)[0]
        return major.isdigit() and int(major) >= 26

    def render_gradle(self, *, modid: str = "generated_mod", version: str = "1.0.0") -> str:
        """Render Gradle from the provider-bound target model."""
        lines: list[str] = []
        unobfuscated = self._unobfuscated_minecraft(self.target.minecraft_version)

        lines.append("plugins {")
        if self.target.loader == "fabric":
            loom_ver = self.target.plugin_versions["fabric-loom"]
            loom_plugin = (
                "net.fabricmc.fabric-loom"
                if unobfuscated
                else "net.fabricmc.fabric-loom-remap"
            )
            lines.append(f'    id "{loom_plugin}" version "{loom_ver}"')
        lines.append('    id "maven-publish"')
        lines.append('    id "java"')
        lines.append("}")
        lines.append("")

        lines.append(f'version = "{version}"')
        lines.append(f'group = "ai.minecraft.{modid}"')
        lines.append(f'base.archivesName = "{modid}"')
        lines.append("")

        lines.append("repositories {")
        lines.append("    mavenCentral()")
        if self.target.loader == "fabric":
            lines.append('    maven { url = "https://maven.fabricmc.net/" }')
        for repo in self.repositories:
            lines.append(f'    maven {{ url = "{repo.url}" }}')
        lines.append("}")
        lines.append("")

        lines.append("dependencies {")
        if self.target.loader == "fabric":
            implementation = "implementation" if unobfuscated else "modImplementation"
            lines.append(
                f'    minecraft "com.mojang:minecraft:{self.target.minecraft_version}"'
            )
            if not unobfuscated:
                if self.target.mappings_kind == "mojang":
                    lines.append("    mappings loom.officialMojangMappings()")
                elif self.target.mappings_kind == "yarn":
                    lines.append(
                        '    mappings '
                        f'"net.fabricmc:yarn:{self.target.mappings}:v2"'
                    )
                else:
                    raise ValueError(
                        f"Unsupported mappings kind: {self.target.mappings_kind!r}"
                    )
            lines.append(
                f'    {implementation} "net.fabricmc:fabric-loader:{self.target.loader_version}"'
            )
            lines.append(
                f'    {implementation} '
                f'"net.fabricmc.fabric-api:fabric-api:{self.target.platform_api_version}"'
            )

        for dep in self.dependencies:
            lines.append(f'    {dep.configuration} "{dep.coordinate}"')

        lines.append('    testImplementation "org.junit.jupiter:junit-jupiter:5.10.2"')
        lines.append("}")
        lines.append("")

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
