from __future__ import annotations

import json
import re
import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .generator import _arena_path_length
from .spec import ContentKind, ModSpec


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    status: str
    checks_run: int
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "checks_run": self.checks_run,
            "findings": [asdict(finding) for finding in self.findings],
        }


class ProjectValidator:
    FORBIDDEN_JAVA = (
        "Runtime.getRuntime(",
        "ProcessBuilder(",
        "java.net.",
        "System.setSecurityManager",
        "Files.delete(",
        "Files.deleteIfExists(",
    )

    def validate(self, root: Path, spec: ModSpec) -> ValidationReport:
        root = root.resolve()
        findings: list[Finding] = []
        checks = 0

        for path in sorted(root.rglob("*")):
            checks += 1
            if path.is_symlink():
                findings.append(
                    Finding("SYMLINK", "error", self._rel(root, path), "Symlinks are not allowed.")
                )
                continue
            try:
                path.relative_to(root)
            except ValueError:
                findings.append(
                    Finding("PATH_ESCAPE", "error", str(path), "Path escaped the staging root.")
                )
            if path.is_file() and path.stat().st_size > 4 * 1024 * 1024:
                findings.append(
                    Finding(
                        "FILE_TOO_LARGE",
                        "error",
                        self._rel(root, path),
                        "Generated source/resource files may not exceed 4 MiB.",
                    )
                )

        json_files = list(root.rglob("*.json")) + list(root.rglob("*.mcmeta"))
        for path in sorted(json_files):
            checks += 1
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                findings.append(
                    Finding("INVALID_JSON", "error", self._rel(root, path), str(exc))
                )

        java_files = sorted(root.rglob("*.java"))
        if not java_files:
            findings.append(Finding("NO_JAVA", "error", ".", "No Java sources were generated."))
        for path in java_files:
            checks += 1
            text = path.read_text(encoding="utf-8")
            for token in self.FORBIDDEN_JAVA:
                if token in text:
                    findings.append(
                        Finding(
                            "FORBIDDEN_JAVA_API",
                            "error",
                            self._rel(root, path),
                            f"Generated source contains forbidden API token {token!r}.",
                        )
                    )

        en_path = root / f"src/main/resources/assets/{spec.mod_id}/lang/en_us.json"
        ko_path = root / f"src/main/resources/assets/{spec.mod_id}/lang/ko_kr.json"
        en = self._load_json(en_path, findings, root)
        ko = self._load_json(ko_path, findings, root)
        checks += 2

        for content in spec.contents:
            prefix = "item" if content.kind is ContentKind.ITEM else "block"
            translation_key = f"{prefix}.{spec.mod_id}.{content.content_id}"
            for locale, translations, path in (("en_us", en, en_path), ("ko_kr", ko, ko_path)):
                checks += 1
                if translation_key not in translations:
                    findings.append(
                        Finding(
                            "MISSING_TRANSLATION",
                            "error",
                            self._rel(root, path),
                            f"{locale} is missing {translation_key}.",
                        )
                    )

            if content.kind is ContentKind.ITEM:
                texture = root / (
                    f"src/main/resources/assets/{spec.mod_id}/textures/item/"
                    f"{content.content_id}.png"
                )
                model = root / (
                    f"src/main/resources/assets/{spec.mod_id}/models/item/"
                    f"{content.content_id}.json"
                )
                required = (texture, model)
            else:
                required = (
                    root
                    / f"src/main/resources/assets/{spec.mod_id}/textures/block/{content.content_id}.png",
                    root
                    / f"src/main/resources/assets/{spec.mod_id}/models/block/{content.content_id}.json",
                    root
                    / f"src/main/resources/assets/{spec.mod_id}/models/item/{content.content_id}.json",
                    root
                    / f"src/main/resources/assets/{spec.mod_id}/blockstates/{content.content_id}.json",
                    root
                    / f"src/main/resources/data/{spec.mod_id}/loot_tables/blocks/{content.content_id}.json",
                )
            for path in required:
                checks += 1
                if not path.is_file():
                    findings.append(
                        Finding(
                            "MISSING_RESOURCE",
                            "error",
                            self._rel(root, path),
                            f"Required resource is missing for {content.content_id}.",
                        )
                    )
            texture_paths = [path for path in required if path.suffix == ".png"]
            for texture in texture_paths:
                checks += 1
                self._validate_png(root, texture, findings)

            if content.recipe:
                recipe_path = (
                    root
                    / f"src/main/resources/data/{spec.mod_id}/recipes/{content.content_id}.json"
                )
                checks += 1
                recipe = self._load_json(recipe_path, findings, root)
                result = recipe.get("result", {}) if isinstance(recipe, dict) else {}
                if not isinstance(result, dict) or result.get("item") != (
                    f"{spec.mod_id}:{content.content_id}"
                ):
                    findings.append(
                        Finding(
                            "BAD_RECIPE_RESULT",
                            "error",
                            self._rel(root, recipe_path),
                            f"Recipe result does not target {content.content_id}.",
                        )
                    )

        if spec.boss is not None:
            boss = spec.boss
            package_path = Path(*spec.package_name.split("."))
            main_class = "".join(part.capitalize() for part in spec.mod_id.split("_")) + "Mod"
            boss_class = (
                "".join(part.capitalize() for part in boss.entity_id.split("_"))
                + "ModEntity"
            )
            renderer_class = (
                "".join(part.capitalize() for part in boss.entity_id.split("_"))
                + "ModRenderer"
            )
            required_boss = (
                root
                / f"src/main/resources/assets/{spec.mod_id}/textures/entity/{boss.entity_id}.png",
                root
                / f"src/main/resources/assets/{spec.mod_id}/models/item/{boss.entity_id}_spawn_egg.json",
                root
                / f"src/main/resources/data/{spec.mod_id}/loot_tables/entities/{boss.entity_id}.json",
                root / "src/main/java" / package_path / "entity" / f"{boss_class}.java",
                root / "src/main/java" / package_path / "client" / f"{renderer_class}.java",
                root / "src/main/java" / package_path / "client" / f"{main_class}Client.java",
                root / f".minecraft_ai/art_sources/{boss.entity_id}.bbmodel",
                root / f".minecraft_ai/art_sources/{boss.entity_id}.obj",
                root / f".minecraft_ai/art_sources/{boss.entity_id}.mtl",
            )
            for path in required_boss:
                checks += 1
                if not path.is_file():
                    findings.append(
                        Finding(
                            "MISSING_BOSS_ASSET",
                            "error",
                            self._rel(root, path),
                            f"Boss asset is missing for {boss.entity_id}.",
                        )
                    )
            entity_texture = required_boss[0]
            checks += 1
            self._validate_png(root, entity_texture, findings)
            bbmodel = self._load_json(required_boss[6], findings, root)
            checks += 1
            if bbmodel.get("model_identifier") != f"{spec.mod_id}:{boss.entity_id}":
                findings.append(
                    Finding(
                        "BAD_BBMODEL",
                        "error",
                        self._rel(root, required_boss[6]),
                        "Blockbench model identifier does not match the boss.",
                    )
                )
            obj_text = (
                required_boss[7].read_text(encoding="utf-8")
                if required_boss[7].is_file()
                else ""
            )
            checks += 2
            if len(re.findall(r"^v\s", obj_text, re.MULTILINE)) < 8:
                findings.append(
                    Finding(
                        "BAD_OBJ",
                        "error",
                        self._rel(root, required_boss[7]),
                        "Boss OBJ has too few vertices.",
                    )
                )
            if len(re.findall(r"^f\s", obj_text, re.MULTILINE)) < 6:
                findings.append(
                    Finding(
                        "BAD_OBJ",
                        "error",
                        self._rel(root, required_boss[7]),
                        "Boss OBJ has too few faces.",
                    )
                )
            for locale, translations, path in (("en_us", en, en_path), ("ko_kr", ko, ko_path)):
                key = f"entity.{spec.mod_id}.{boss.entity_id}"
                checks += 1
                if key not in translations:
                    findings.append(
                        Finding(
                            "MISSING_BOSS_TRANSLATION",
                            "error",
                            self._rel(root, path),
                            f"{locale} is missing {key}.",
                        )
                    )

        if spec.arena is not None:
            arena = spec.arena
            function_path = (
                root
                / f"src/main/resources/data/{spec.mod_id}/functions/build_{arena.arena_id}.mcfunction"
            )
            world_ir_path = root / f".minecraft_ai/world/{arena.arena_id}.world_design.json"
            preview_path = root / f".minecraft_ai/world/{arena.arena_id}_preview.png"
            for path in (function_path, world_ir_path, preview_path):
                checks += 1
                if not path.is_file():
                    findings.append(
                        Finding(
                            "MISSING_ARENA_ASSET",
                            "error",
                            self._rel(root, path),
                            f"Arena artifact is missing for {arena.arena_id}.",
                        )
                    )
            function_text = (
                function_path.read_text(encoding="utf-8")
                if function_path.is_file()
                else ""
            )
            checks += 1
            if spec.boss is not None:
                checks += 1
                expected_summon = f"summon {spec.mod_id}:{spec.boss.entity_id}"
                if expected_summon not in function_text:
                    findings.append(
                        Finding(
                            "ARENA_BOSS_MISSING",
                            "error",
                            self._rel(root, function_path),
                            "Arena function does not summon the explicitly requested boss.",
                        )
                    )
            else:
                checks += 1
                if any(
                    line.strip().startswith("summon ")
                    for line in function_text.splitlines()
                ):
                    findings.append(
                        Finding(
                            "ARENA_UNAPPROVED_SUMMON",
                            "error",
                            self._rel(root, function_path),
                            "A boss-free arena may not contain a summon command.",
                        )
                    )
            if len(function_text.splitlines()) > 64:
                findings.append(
                    Finding(
                        "ARENA_COMMAND_BUDGET",
                        "error",
                        self._rel(root, function_path),
                        "Arena function exceeds the bounded command budget.",
                    )
                )
            expected_commands = {
                (
                    f"fill ~-{arena.radius} ~-1 ~-{arena.radius} "
                    f"~{arena.radius} ~-1 ~{arena.radius} {arena.floor_block}"
                ),
                (
                    f"fill ~-{arena.radius} ~ ~-{arena.radius} "
                    f"~{arena.radius} ~{arena.wall_height} ~-{arena.radius} "
                    f"{arena.accent_block}"
                ),
                (
                    f"fill ~-{arena.radius} ~ ~{arena.radius} "
                    f"~{arena.radius} ~{arena.wall_height} ~{arena.radius} "
                    f"{arena.accent_block}"
                ),
                (
                    f"fill ~-{arena.radius} ~ ~-{arena.radius - 1} "
                    f"~-{arena.radius} ~{arena.wall_height} ~{arena.radius - 1} "
                    f"{arena.accent_block}"
                ),
                (
                    f"fill ~{arena.radius} ~ ~-{arena.radius - 1} "
                    f"~{arena.radius} ~{arena.wall_height} ~{arena.radius - 1} "
                    f"{arena.accent_block}"
                ),
                f"fill ~-2 ~ ~-{arena.radius} ~2 ~2 ~-{arena.radius} air",
            }
            actual_commands = set(function_text.splitlines())
            for command in sorted(expected_commands):
                checks += 1
                if command not in actual_commands:
                    findings.append(
                        Finding(
                            "ARENA_GEOMETRY_MISMATCH",
                            "error",
                            self._rel(root, function_path),
                            f"Required bounded arena command is missing: {command}",
                        )
                    )
            world_ir = self._load_json(world_ir_path, findings, root)
            checks += 1
            navigation = world_ir.get("navigation", {})
            expected_path_length = _arena_path_length(arena.radius)
            center_zone = "boss_area" if spec.boss is not None else "map_center"
            verification = (
                navigation.get("verification", {})
                if isinstance(navigation, dict)
                else {}
            )
            if (
                not isinstance(navigation, dict)
                or navigation.get("critical_path_verified") is not True
                or ["entry", center_zone] not in navigation.get("required_paths", [])
                or navigation.get("minimum_door_width") != 5
                or not isinstance(verification, dict)
                or verification.get("method") != "deterministic_grid_bfs"
                or verification.get("result") is not True
                or verification.get("path_length_blocks") != expected_path_length
                or expected_path_length is None
            ):
                findings.append(
                    Finding(
                        "ARENA_PATH_INVALID",
                        "error",
                        self._rel(root, world_ir_path),
                        (
                            "WorldDesignIR does not prove the entry-to-boss critical path."
                            if spec.boss is not None
                            else "WorldDesignIR does not prove the entry-to-center critical path."
                        ),
                    )
                )
            self._validate_png(root, preview_path, findings)

        fabric_path = root / "src/main/resources/fabric.mod.json"
        fabric = self._load_json(fabric_path, findings, root)
        checks += 1
        if fabric.get("id") != spec.mod_id:
            findings.append(
                Finding("BAD_MOD_ID", "error", self._rel(root, fabric_path), "fabric.mod.json id mismatch.")
            )
        if fabric.get("version") != "${version}":
            findings.append(
                Finding(
                    "UNPINNED_VERSION",
                    "error",
                    self._rel(root, fabric_path),
                    "fabric.mod.json must use the Gradle-expanded version.",
                )
            )
        if spec.boss is None:
            entrypoints = fabric.get("entrypoints")
            checks += 1
            if isinstance(entrypoints, dict) and "client" in entrypoints:
                findings.append(
                    Finding(
                        "UNAPPROVED_BOSS_ARTIFACT",
                        "error",
                        self._rel(root, fabric_path),
                        "A boss-free project may not declare the generated client entrypoint.",
                    )
                )
            forbidden_roots = (
                root / f"src/main/java/{spec.package_name.replace('.', '/')}/entity",
                root / f"src/main/java/{spec.package_name.replace('.', '/')}/client",
                root / f"src/main/resources/assets/{spec.mod_id}/textures/entity",
                root / f"src/main/resources/data/{spec.mod_id}/loot_tables/entities",
            )
            for forbidden_root in forbidden_roots:
                checks += 1
                if forbidden_root.exists() and any(
                    path.is_file() for path in forbidden_root.rglob("*")
                ):
                    findings.append(
                        Finding(
                            "UNAPPROVED_BOSS_ARTIFACT",
                            "error",
                            self._rel(root, forbidden_root),
                            "Boss/entity runtime artifacts exist without an approved boss.",
                        )
                    )
            spawn_egg_root = (
                root
                / f"src/main/resources/assets/{spec.mod_id}/models/item"
            )
            checks += 1
            if spawn_egg_root.is_dir() and any(
                spawn_egg_root.glob("*_spawn_egg.json")
            ):
                findings.append(
                    Finding(
                        "UNAPPROVED_BOSS_ARTIFACT",
                        "error",
                        self._rel(root, spawn_egg_root),
                        "A spawn-egg model exists without an approved boss.",
                    )
                )

        status = "PASS" if not any(item.severity == "error" for item in findings) else "FAIL"
        return ValidationReport(status=status, checks_run=checks, findings=tuple(findings))

    @staticmethod
    def _load_json(path: Path, findings: list[Finding], root: Path) -> dict[str, object]:
        if not path.is_file():
            findings.append(Finding("MISSING_JSON", "error", str(path), "Required JSON is missing."))
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.append(Finding("INVALID_JSON", "error", str(path), str(exc)))
            return {}
        if not isinstance(value, dict):
            findings.append(Finding("JSON_NOT_OBJECT", "error", str(path), "Expected a JSON object."))
            return {}
        return value

    @staticmethod
    def _validate_png(root: Path, path: Path, findings: list[Finding]) -> None:
        if not path.is_file():
            return
        data = path.read_bytes()
        relative = str(path.relative_to(root)).replace("\\", "/")
        if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            findings.append(Finding("INVALID_PNG", "error", relative, "Invalid PNG signature."))
            return
        width, height = struct.unpack(">II", data[16:24])
        if (width, height) not in {(16, 16), (32, 32), (64, 64)}:
            findings.append(
                Finding(
                    "BAD_TEXTURE_SIZE",
                    "error",
                    relative,
                    f"Texture is {width}x{height}; expected 16x16, 32x32, or 64x64.",
                )
            )

    @staticmethod
    def _rel(root: Path, path: Path) -> str:
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(path)


def validate_jar(jar_path: Path, spec: ModSpec) -> ValidationReport:
    findings: list[Finding] = []
    checks = 0
    if not jar_path.is_file():
        return ValidationReport(
            status="FAIL",
            checks_run=1,
            findings=(Finding("JAR_MISSING", "error", str(jar_path), "Built JAR is missing."),),
        )
    checks += 1
    if not zipfile.is_zipfile(jar_path):
        return ValidationReport(
            status="FAIL",
            checks_run=checks + 1,
            findings=(Finding("NOT_A_JAR", "error", str(jar_path), "File is not a ZIP/JAR archive."),),
        )

    with zipfile.ZipFile(jar_path) as archive:
        raw_names = archive.namelist()
        names = set(raw_names)
        checks += 1
        if len(names) != len(raw_names):
            findings.append(
                Finding("JAR_DUPLICATE_ENTRY", "error", str(jar_path), "JAR has duplicate entries.")
            )
        for name in raw_names:
            normalized = name.replace("\\", "/")
            if normalized.startswith("/") or ".." in Path(normalized).parts:
                findings.append(
                    Finding("JAR_UNSAFE_PATH", "error", name, "Unsafe path found in JAR.")
                )
        checks += 1
        corrupt = archive.testzip()
        if corrupt is not None:
            findings.append(
                Finding("JAR_BAD_CRC", "error", corrupt, "JAR entry failed its CRC check.")
            )

        metadata: dict[str, object] = {}
        if "fabric.mod.json" not in names:
            findings.append(
                Finding("JAR_NO_METADATA", "error", str(jar_path), "fabric.mod.json is absent.")
            )
        else:
            checks += 1
            try:
                metadata = json.loads(archive.read("fabric.mod.json").decode("utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("fabric.mod.json is not an object")
                expected_metadata = {
                    "id": spec.mod_id,
                    "version": spec.version,
                    "environment": "*",
                }
                for key, expected in expected_metadata.items():
                    if metadata.get(key) != expected:
                        findings.append(
                            Finding(
                                "JAR_BAD_METADATA",
                                "error",
                                "fabric.mod.json",
                                f"{key} must equal {expected!r}.",
                            )
                        )
                depends = metadata.get("depends")
                expected_depends = {
                    "fabricloader": f">={spec.platform.fabric_loader}",
                    "minecraft": f"~{spec.platform.minecraft_version}",
                    "java": ">=17",
                    "fabric-api": f">={spec.platform.fabric_api}",
                }
                if not isinstance(depends, dict):
                    findings.append(
                        Finding(
                            "JAR_BAD_DEPENDS",
                            "error",
                            "fabric.mod.json",
                            "depends must be an object.",
                        )
                    )
                else:
                    for dependency, constraint in expected_depends.items():
                        if depends.get(dependency) != constraint:
                            findings.append(
                                Finding(
                                    "JAR_BAD_DEPENDS",
                                    "error",
                                    "fabric.mod.json",
                                    f"{dependency} must equal {constraint!r}.",
                                )
                            )
            except Exception as exc:
                findings.append(
                    Finding("JAR_BAD_METADATA", "error", str(jar_path), str(exc))
                )

        java_root = spec.package_name.replace(".", "/")
        main_class = "".join(part.capitalize() for part in spec.mod_id.split("_")) + "Mod"
        checks += 1
        if not any(name.endswith(".class") for name in names):
            findings.append(
                Finding("JAR_NO_CLASSES", "error", str(jar_path), "No class files found.")
            )
        expected_classes = {
            f"{java_root}/{main_class}.class",
            f"{java_root}/{main_class}GameTests.class",
            f"{java_root}/GeneratedContent.class",
        }
        expected_entrypoints: dict[str, str] = {
            "main": f"{spec.package_name}.{main_class}",
            "fabric-gametest": f"{spec.package_name}.{main_class}GameTests",
        }
        if spec.boss is not None:
            boss_class = (
                "".join(part.capitalize() for part in spec.boss.entity_id.split("_")) + "Mod"
            )
            expected_classes.update(
                {
                    f"{java_root}/entity/{boss_class}Entity.class",
                    f"{java_root}/client/{boss_class}Renderer.class",
                    f"{java_root}/client/{main_class}Client.class",
                }
            )
            expected_entrypoints["client"] = (
                f"{spec.package_name}.client.{main_class}Client"
            )

        checks += len(expected_classes)
        for class_name in sorted(expected_classes):
            if class_name not in names:
                findings.append(
                    Finding(
                        "JAR_CLASS_MISSING",
                        "error",
                        class_name,
                        "Required compiled entrypoint/support class is absent.",
                    )
                )
                continue
            if not archive.read(class_name).startswith(b"\xCA\xFE\xBA\xBE"):
                findings.append(
                    Finding(
                        "JAR_BAD_CLASS",
                        "error",
                        class_name,
                        "Class entry does not have the JVM class-file magic.",
                    )
                )

        entrypoints = metadata.get("entrypoints") if isinstance(metadata, dict) else None
        checks += len(expected_entrypoints)
        if not isinstance(entrypoints, dict):
            findings.append(
                Finding("JAR_BAD_ENTRYPOINTS", "error", "fabric.mod.json", "entrypoints missing.")
            )
        else:
            for group, expected_class in expected_entrypoints.items():
                values = entrypoints.get(group)
                if not isinstance(values, list) or expected_class not in values:
                    findings.append(
                        Finding(
                            "JAR_BAD_ENTRYPOINTS",
                            "error",
                            "fabric.mod.json",
                            f"{group} must include {expected_class}.",
                        )
                    )
        if spec.boss is None:
            checks += 1
            if isinstance(entrypoints, dict) and "client" in entrypoints:
                findings.append(
                    Finding(
                        "JAR_UNAPPROVED_BOSS_ARTIFACT",
                        "error",
                        "fabric.mod.json",
                        "A boss-free JAR may not declare the generated client entrypoint.",
                    )
                )
            forbidden_names = {
                name
                for name in names
                if (
                    name.startswith(f"{java_root}/entity/")
                    or name.startswith(f"{java_root}/client/")
                    or name.startswith(f"assets/{spec.mod_id}/textures/entity/")
                    or name.startswith(f"data/{spec.mod_id}/loot_tables/entities/")
                    or name.endswith("_spawn_egg.json")
                )
            }
            checks += 1
            for forbidden_name in sorted(forbidden_names):
                findings.append(
                    Finding(
                        "JAR_UNAPPROVED_BOSS_ARTIFACT",
                        "error",
                        forbidden_name,
                        "Boss/entity runtime artifact exists without an approved boss.",
                    )
                )
            for function_name in sorted(
                name for name in names if name.endswith(".mcfunction")
            ):
                checks += 1
                try:
                    function_text = archive.read(function_name).decode("utf-8")
                except (KeyError, UnicodeDecodeError):
                    continue
                if any(
                    line.strip().startswith("summon ")
                    for line in function_text.splitlines()
                ):
                    findings.append(
                        Finding(
                            "JAR_UNAPPROVED_SUMMON",
                            "error",
                            function_name,
                            "A boss-free JAR may not contain a summon command.",
                        )
                    )

        required_resources = {
            f"assets/{spec.mod_id}/lang/en_us.json",
            f"assets/{spec.mod_id}/lang/ko_kr.json",
        }
        for content in spec.contents:
            if content.kind is ContentKind.ITEM:
                required_resources.update(
                    {
                        f"assets/{spec.mod_id}/models/item/{content.content_id}.json",
                        f"assets/{spec.mod_id}/textures/item/{content.content_id}.png",
                    }
                )
            else:
                required_resources.update(
                    {
                        f"assets/{spec.mod_id}/blockstates/{content.content_id}.json",
                        f"assets/{spec.mod_id}/models/block/{content.content_id}.json",
                        f"assets/{spec.mod_id}/models/item/{content.content_id}.json",
                        f"assets/{spec.mod_id}/textures/block/{content.content_id}.png",
                        f"data/{spec.mod_id}/loot_tables/blocks/{content.content_id}.json",
                    }
                )
            if content.recipe:
                required_resources.add(
                    f"data/{spec.mod_id}/recipes/{content.content_id}.json"
                )
        if spec.boss is not None:
            required_resources.update(
                {
                    f"assets/{spec.mod_id}/textures/entity/{spec.boss.entity_id}.png",
                    f"assets/{spec.mod_id}/models/item/{spec.boss.entity_id}_spawn_egg.json",
                    f"data/{spec.mod_id}/loot_tables/entities/{spec.boss.entity_id}.json",
                }
            )
        if spec.arena is not None:
            required_resources.add(
                f"data/{spec.mod_id}/functions/build_{spec.arena.arena_id}.mcfunction"
            )

        checks += len(required_resources)
        for required in sorted(required_resources):
            if required not in names:
                findings.append(
                    Finding(
                        "JAR_RESOURCE_MISSING",
                        "error",
                        required,
                        "Approved runtime resource is absent from JAR.",
                    )
                )

    status = "PASS" if not any(item.severity == "error" for item in findings) else "FAIL"
    return ValidationReport(status=status, checks_run=checks, findings=tuple(findings))
