from __future__ import annotations

"""Component-catalog extraction from immutable project inventory evidence."""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from .project_inventory import (
    COMPONENT_CATALOG_SCHEMA,
    ComponentCatalog,
    ComponentRecord,
    DependencyRecord,
    EvidenceLocator,
    ModMetadata,
    ProjectInventoryError,
    ProjectModule,
    ProjectTarget,
    _IMPORT,
    _JAVA_FIELD,
    _JAVA_METHOD,
    _JAVA_TYPE,
    _KOTLIN_CALLABLE,
    _KOTLIN_TYPE,
    _PACKAGE,
    _RESOURCE_ID,
    _TWO_PART_IDENTIFIER,
    _ScannedFile,
    _component_id,
    _evidence,
    _is_text,
    _module_for_path,
    _read_bound_text,
    _source_root_for_path,
    canonical_sha256,
)


def _resource_identity(path: str) -> tuple[tuple[str, ...], str, str]:
    parts = path.split("/")
    marker_index = next(
        (index for index, part in enumerate(parts) if part in {"assets", "data"}),
        -1,
    )
    if marker_index < 0 or len(parts) <= marker_index + 2:
        return (), "", ""
    scope = parts[marker_index]
    namespace = parts[marker_index + 1].casefold()
    relative_parts = parts[marker_index + 2 :]
    filename = relative_parts[-1]
    suffix = Path(filename).suffix
    if suffix:
        relative_parts[-1] = filename[: -len(suffix)]
    relative = "/".join(relative_parts).casefold()
    category, _, remainder = relative.partition("/")
    kind_map = {
        "advancements": "advancement",
        "blockstates": "blockstate",
        "functions": "function",
        "lang": "lang",
        "loot_tables": "loot_table",
        "models": "model",
        "recipes": "recipe",
        "tags": "tag",
        "textures": "texture",
        "worldgen": "worldgen",
    }
    kind = kind_map.get(category, "resource")
    semantic_path = remainder if remainder else category
    provides = {
        f"resource:{scope}:{namespace}:{relative}",
        f"{kind}:{namespace}:{semantic_path}",
    }
    return tuple(sorted(provides)), namespace, kind


def _resource_references(text: str) -> tuple[str, ...]:
    references = {match.group(1).casefold() for match in _RESOURCE_ID.finditer(text)}
    references.update(
        f"{match.group(1).casefold()}:{match.group(2).casefold()}"
        for match in _TWO_PART_IDENTIFIER.finditer(text)
    )
    for match in re.finditer(
        r"[\"']parent[\"']\s*:\s*[\"']([^\"':]+/[^\"']+)[\"']",
        text,
    ):
        references.add("minecraft:" + match.group(1).casefold())
    return tuple(sorted(references))


def _snake_case_identifier(value: str) -> str:
    """Return one deterministic semantic token; this is not fuzzy matching."""

    separated = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", separated)
    separated = re.sub(r"[^A-Za-z0-9]+", "_", separated).strip("_").casefold()
    return re.sub(r"_+", "_", separated)


def _capability_aliases(name: str) -> tuple[str, ...]:
    """Derive exact host-owned aliases from a symbol or logical resource name."""

    candidates: list[str] = []
    if "#" in name:
        owner, member = name.rsplit("#", 1)
        owner_name = owner.rsplit(".", 1)[-1]
        candidates.extend((member, owner_name + "_" + member))
    else:
        tail = name.rsplit(":", 1)[-1].rsplit("/", 1)[-1].rsplit(".", 1)[-1]
        candidates.append(tail)
    aliases = {
        "capability:" + normalized
        for candidate in candidates
        if (normalized := _snake_case_identifier(candidate))
    }
    return tuple(sorted(aliases))


def _test_capability_aliases(name: str) -> tuple[str, ...]:
    """Bind conventional test owners to the exact subject name they verify."""

    owner = name.split("#", 1)[0]
    prefix, separator, simple = owner.rpartition(".")
    subject = re.sub(r"(?:GameTests?|Tests?)$", "", simple)
    if not subject or subject == simple:
        return ()
    qualified = f"{prefix}{separator}{subject}" if prefix else subject
    return _capability_aliases(qualified)


def component_record(
    *,
    kind: str,
    name: str,
    evidence: EvidenceLocator,
    module_id: str,
    source_set: str,
    side: str,
    target: ProjectTarget,
    provides: Iterable[str],
    requires: Iterable[str],
    license_refs: tuple[str, ...],
) -> ComponentRecord:
    locator = evidence.locator
    return ComponentRecord(
        component_id=_component_id(
            kind=kind,
            name=name,
            locator=locator,
            module_id=module_id,
        ),
        kind=kind,
        name=name,
        locator=locator,
        content_sha256=evidence.sha256,
        module_id=module_id,
        source_set=source_set,
        side=side,
        minecraft_versions=target.minecraft_versions,
        loaders=target.loaders,
        provides=tuple(
            sorted({str(item).strip() for item in provides if str(item).strip()})
        ),
        requires=tuple(
            sorted({str(item).strip() for item in requires if str(item).strip()})
        ),
        evidence=(evidence,),
        license_refs=license_refs,
    )


def _source_location(
    file: _ScannedFile,
    module_paths: Mapping[str, str],
) -> tuple[str, str, str]:
    module_id = _module_for_path(file.path, module_paths)
    root = _source_root_for_path(module_id, module_paths[module_id], file.path)
    if root is None:
        return module_id, "", "common"
    lowered = root.source_set.casefold()
    side = (
        "client"
        if "client" in lowered
        else "server"
        if "server" in lowered
        else "common"
    )
    return module_id, root.source_set, side


def source_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> tuple[list[ComponentRecord], set[str], set[str]]:
    components: list[ComponentRecord] = []
    namespaces: set[str] = set()
    references: set[str] = set()
    for file in files:
        suffix = Path(file.path).suffix.casefold()
        if suffix not in {".java", ".kt"}:
            continue
        text = _read_bound_text(file)
        package_match = _PACKAGE.search(text)
        package = package_match.group(1) if package_match else ""
        if package:
            namespaces.add(package)
        imports = {"symbol:" + item.rstrip(".*") for item in _IMPORT.findall(text)}
        resource_refs = set(_resource_references(text))
        references.update(resource_refs)
        requires = imports | {"resource_ref:" + item for item in resource_refs}
        module_id, source_set, side = _source_location(file, module_paths)
        is_test = (
            "test" in source_set.casefold()
            or Path(file.path).stem.endswith("Test")
        )
        owner = ""
        found = False
        lines = text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            type_match = (
                _JAVA_TYPE.search(line)
                if suffix == ".java"
                else _KOTLIN_TYPE.search(line)
            )
            if type_match:
                simple_name = type_match.group("name")
                owner = f"{package}.{simple_name}" if package else simple_name
                evidence = _evidence(file, line_number, line_number)
                symbol = "symbol:" + owner
                provides = (
                    {symbol, "namespace:" + package}
                    if package
                    else {symbol}
                )
                provides.update(_capability_aliases(owner))
                if is_test:
                    provides.add("test:" + owner)
                    provides.update(_test_capability_aliases(owner))
                components.append(
                    component_record(
                        kind="test" if is_test else "symbol",
                        name=owner,
                        evidence=evidence,
                        module_id=module_id,
                        source_set=source_set,
                        side=side,
                        target=target,
                        provides=provides,
                        requires=requires,
                        license_refs=license_refs,
                    )
                )
                found = True
                continue

            callable_match = (
                _JAVA_METHOD.match(line)
                if suffix == ".java"
                else _KOTLIN_CALLABLE.match(line)
            )
            if (
                callable_match
                and callable_match.group("name")
                not in {"if", "for", "while", "switch", "catch"}
            ):
                callable_name = callable_match.group("name")
                qualified = (
                    owner or package or Path(file.path).stem
                ) + "#" + callable_name
                evidence = _evidence(file, line_number, line_number)
                provides = {
                    "symbol:" + qualified,
                    *_capability_aliases(qualified),
                }
                if is_test or "@Test" in line or "@GameTest" in line:
                    provides.add("test:" + qualified)
                    provides.update(_test_capability_aliases(qualified))
                components.append(
                    component_record(
                        kind="test" if is_test else "symbol",
                        name=qualified,
                        evidence=evidence,
                        module_id=module_id,
                        source_set=source_set,
                        side=side,
                        target=target,
                        provides=provides,
                        requires=requires,
                        license_refs=license_refs,
                    )
                )
                found = True
                continue

            if suffix == ".java":
                field_match = _JAVA_FIELD.match(line)
                if field_match:
                    field_name = field_match.group("name")
                    qualified = (
                        owner or package or Path(file.path).stem
                    ) + "#" + field_name
                    components.append(
                        component_record(
                            kind="symbol",
                            name=qualified,
                            evidence=_evidence(file, line_number, line_number),
                            module_id=module_id,
                            source_set=source_set,
                            side=side,
                            target=target,
                            provides={
                                "symbol:" + qualified,
                                *_capability_aliases(qualified),
                            },
                            requires=requires,
                            license_refs=license_refs,
                        )
                    )
                    found = True

        if not found:
            name = (
                f"{package}.{Path(file.path).stem}"
                if package
                else Path(file.path).stem
            )
            components.append(
                component_record(
                    kind="test" if is_test else "symbol",
                    name=name,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set=source_set,
                    side=side,
                    target=target,
                    provides={
                        ("test-file:" if is_test else "source-file:") + file.path,
                        *_capability_aliases(name),
                        *(_test_capability_aliases(name) if is_test else ()),
                    },
                    requires=requires,
                    license_refs=license_refs,
                )
            )
    return components, namespaces, references


def resource_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> tuple[list[ComponentRecord], set[str], set[str], set[str]]:
    components: list[ComponentRecord] = []
    namespaces: set[str] = set()
    logical_ids: set[str] = set()
    references: set[str] = set()
    for file in files:
        provides, namespace, kind = _resource_identity(file.path)
        if not provides:
            continue
        namespaces.add(namespace)
        logical_ids.update(provides)
        semantic_provides = {*provides}
        for logical_id in provides:
            semantic_provides.update(_capability_aliases(logical_id))
        found_references: tuple[str, ...] = ()
        if _is_text(file):
            found_references = _resource_references(_read_bound_text(file))
            references.update(found_references)
        module_id, source_set, side = _source_location(file, module_paths)
        components.append(
            component_record(
                kind="resource",
                name=provides[0],
                evidence=_evidence(file),
                module_id=module_id,
                source_set=source_set,
                side=side,
                target=target,
                provides=semantic_provides,
                requires={"resource_ref:" + item for item in found_references},
                license_refs=license_refs,
            )
        )
    return components, namespaces, logical_ids, references


def config_components(
    files: Sequence[_ScannedFile],
    module_paths: Mapping[str, str],
    modules: Sequence[ProjectModule],
    dependencies: Sequence[DependencyRecord],
    metadata: Sequence[ModMetadata],
    target: ProjectTarget,
    license_refs: tuple[str, ...],
) -> list[ComponentRecord]:
    components: list[ComponentRecord] = []
    dependencies_by_module: dict[str, list[DependencyRecord]] = {}
    for dependency in dependencies:
        dependencies_by_module.setdefault(dependency.module_id, []).append(dependency)
        components.append(
            component_record(
                kind="dependency",
                name=dependency.coordinate,
                evidence=dependency.evidence,
                module_id=dependency.module_id,
                source_set="",
                side="build",
                target=target,
                provides={"dependency:" + dependency.coordinate},
                requires=(
                    {"module:" + dependency.project_path}
                    if dependency.project_path
                    else ()
                ),
                license_refs=license_refs,
            )
        )

    modules_by_id = {module.module_id: module for module in modules}
    metadata_by_path = {record.path: record for record in metadata}
    for file in files:
        name = Path(file.path).name.casefold()
        module_id = _module_for_path(file.path, module_paths)
        module = modules_by_id.get(module_id)
        if name in {
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
            "gradle.properties",
            "libs.versions.toml",
        } or file.path.casefold().endswith(
            "gradle/wrapper/gradle-wrapper.properties"
        ):
            provides = {"build_config:" + file.path, "module:" + module_id}
            if module is not None:
                provides.update(
                    "source_set:" + module_id + ":" + item
                    for item in module.source_sets
                )
                provides.update(
                    "generated_resources:" + item
                    for item in module.generated_resource_roots
                )
            components.append(
                component_record(
                    kind="build_config",
                    name=file.path,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set="",
                    side="build",
                    target=target,
                    provides=provides,
                    requires={
                        "dependency:" + item.coordinate
                        for item in dependencies_by_module.get(module_id, [])
                    },
                    license_refs=license_refs,
                )
            )

        record = metadata_by_path.get(file.path)
        if record is not None:
            provides = {
                f"metadata:{record.loader}:{record.mod_id}",
                "namespace:" + record.mod_id,
            }
            provides.update(
                f"entrypoint:{entry.loader}:{entry.group}:{entry.value}"
                for entry in record.entrypoints
            )
            requires = {"symbol:" + entry.value for entry in record.entrypoints}
            requires.update(
                "dependency:" + dependency.coordinate
                for dependency in dependencies
                if dependency.dependency_id in record.dependency_ids
            )
            components.append(
                component_record(
                    kind="build_config",
                    name=file.path,
                    evidence=record.evidence,
                    module_id=module_id,
                    source_set="resources",
                    side=record.environment or "common",
                    target=target,
                    provides=provides,
                    requires=requires,
                    license_refs=license_refs,
                )
            )

        lowered_path = file.path.casefold()
        release_name = name in {
            "changelog.md",
            "jenkinsfile",
            "license",
            "license.md",
            "license.txt",
            "notice",
            "notice.txt",
        }
        release_path = (
            lowered_path.startswith(".github/workflows/")
            or name in {"azure-pipelines.yml", ".gitlab-ci.yml"}
            or "publish" in name
            or "/release" in lowered_path
        )
        if release_name or release_path:
            components.append(
                component_record(
                    kind="release_config",
                    name=file.path,
                    evidence=_evidence(file),
                    module_id=module_id,
                    source_set="",
                    side="build",
                    target=target,
                    provides={"release_config:" + file.path},
                    requires={"module:" + module_id},
                    license_refs=license_refs,
                )
            )
    return components


def _deduplicate_components(
    components: Iterable[ComponentRecord],
) -> tuple[ComponentRecord, ...]:
    result: dict[str, ComponentRecord] = {}
    for component in components:
        prior = result.get(component.component_id)
        if prior is not None and prior != component:
            raise ProjectInventoryError(
                f"Component ID collision: {component.component_id}"
            )
        result[component.component_id] = component
    return tuple(sorted(result.values(), key=lambda item: item.component_id))


def build_component_catalog(
    components: Iterable[ComponentRecord],
) -> ComponentCatalog:
    ordered = _deduplicate_components(components)
    catalog = ComponentCatalog(
        schema_version=COMPONENT_CATALOG_SCHEMA,
        components=ordered,
        catalog_sha256="",
    )
    catalog = replace(
        catalog,
        catalog_sha256=canonical_sha256(
            {
                "schema_version": catalog.schema_version,
                "components": catalog.components,
                "catalog_sha256": "",
            }
        ),
    )
    catalog.validate()
    return catalog


__all__ = [
    "build_component_catalog",
    "component_record",
    "config_components",
    "resource_components",
    "source_components",
]
