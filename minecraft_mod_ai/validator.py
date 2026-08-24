from __future__ import annotations
import json
import re
import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from .complete_spec import CompleteProposal
from .local_ai_sidecar_generator import INTEGRATION_TYPE as LOCAL_AI_SIDECAR_INTEGRATION_TYPE, LocalAiSidecarGenerationError, local_ai_sidecar_manifest_path, local_ai_sidecar_source_path, normalize_local_ai_sidecar_config, render_local_ai_sidecar_manifest, render_local_ai_sidecar_source
from .scale_policy import ScalePolicy
from .spec import ContentKind, ModSpec
from .toolchain_contract import fabric_dependency_predicates

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
        return self.status == 'PASS'

    def to_dict(self) -> dict[str, object]:
        return {'status': self.status, 'checks_run': self.checks_run, 'findings': [asdict(finding) for finding in self.findings]}

class ProjectValidator:
    FORBIDDEN_JAVA = ('Runtime.getRuntime(', 'ProcessBuilder(', 'java.net.', 'System.setSecurityManager', 'Files.delete(', 'Files.deleteIfExists(')

    def __init__(self, *, policy: ScalePolicy | None=None) -> None:
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()

    def validate(self, root: Path, spec: ModSpec) -> ValidationReport:
        root = root.expanduser().resolve()
        findings: list[Finding] = []
        checks = 0
        if not root.is_dir() or root.is_symlink():
            return ValidationReport(status='FAIL', checks_run=1, findings=(Finding('PROJECT_ROOT_INVALID', 'error', str(root), 'Project root must be a regular directory.'),))
        complete = _load_complete_project_proposal(root, spec, findings)
        complete_entity_ids = _complete_entity_ids(complete)
        complete_client_allowed = _complete_client_required(complete)
        reviewed_network_sources = _reviewed_local_ai_sidecar_network_sources(root, spec, complete)
        for path in sorted(root.rglob('*')):
            checks += 1
            relative = self._rel(root, path)
            if path.is_symlink():
                findings.append(Finding('SYMLINK', 'error', relative, 'Symlinks are not allowed.'))
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                findings.append(Finding('PATH_ESCAPE', 'error', relative, 'Path escaped the staging root.'))
                continue
            if path.is_file() and path.stat().st_size > self.policy.max_single_file_bytes:
                findings.append(Finding('FILE_TOO_LARGE', 'error', relative, 'File exceeds MMM_MAX_SINGLE_FILE_BYTES host resource policy.'))
        for path in sorted({*root.rglob('*.json'), *root.rglob('*.mcmeta')}):
            checks += 1
            self._load_json(path, findings, root)
        java_files = sorted(root.rglob('*.java'))
        checks += 1
        if not java_files:
            findings.append(Finding('NO_JAVA', 'error', '.', 'No Java sources were generated.'))
        for path in java_files:
            checks += 1
            text = path.read_text(encoding='utf-8', errors='replace')
            relative = self._rel(root, path)
            for token in self.FORBIDDEN_JAVA:
                if token in text:
                    if token == 'java.net.' and reviewed_network_sources.get(relative) == text:
                        continue
                    findings.append(Finding('FORBIDDEN_JAVA_API', 'error', relative, f'Generated source contains forbidden API token {token!r}.'))
        en_path = root / f'src/main/resources/assets/{spec.mod_id}/lang/en_us.json'
        ko_path = root / f'src/main/resources/assets/{spec.mod_id}/lang/ko_kr.json'
        en = self._load_json(en_path, findings, root)
        ko = self._load_json(ko_path, findings, root)
        checks += 2
        for content in spec.contents:
            checks += self._validate_content(root, spec, content, en, ko, en_path, ko_path, findings)
        if spec.boss is not None:
            checks += self._validate_boss(root, spec, en, ko, en_path, ko_path, findings)
        fabric_path = root / 'src/main/resources/fabric.mod.json'
        fabric = self._load_json(fabric_path, findings, root)
        checks += self._validate_fabric_metadata(root, spec, fabric_path, fabric, complete, findings)
        if spec.boss is None and (not complete_entity_ids) and (not complete_client_allowed):
            checks += self._validate_no_unapproved_client_or_entity(root, spec, fabric, findings)
        if complete is not None:
            checks += self._validate_complete_sources(root, spec, complete, findings)
        status = 'PASS' if not any((item.severity == 'error' for item in findings)) else 'FAIL'
        return ValidationReport(status=status, checks_run=checks, findings=tuple(findings))

    def _validate_content(self, root: Path, spec: ModSpec, content: Any, en: dict[str, Any], ko: dict[str, Any], en_path: Path, ko_path: Path, findings: list[Finding]) -> int:
        checks = 0
        prefix = 'item' if content.kind is ContentKind.ITEM else 'block'
        translation_key = f'{prefix}.{spec.mod_id}.{content.content_id}'
        for locale, translations, path in (('en_us', en, en_path), ('ko_kr', ko, ko_path)):
            checks += 1
            if translation_key not in translations:
                findings.append(Finding('MISSING_TRANSLATION', 'error', self._rel(root, path), f'{locale} is missing {translation_key}.'))
        assets = root / f'src/main/resources/assets/{spec.mod_id}'
        data = root / f'src/main/resources/data/{spec.mod_id}'
        if content.kind is ContentKind.ITEM:
            required = (assets / 'textures/item' / f'{content.content_id}.png', assets / 'models/item' / f'{content.content_id}.json')
        else:
            required = (assets / 'textures/block' / f'{content.content_id}.png', assets / 'models/block' / f'{content.content_id}.json', assets / 'models/item' / f'{content.content_id}.json', assets / 'blockstates' / f'{content.content_id}.json', data / 'loot_tables/blocks' / f'{content.content_id}.json')
        for path in required:
            checks += 1
            if not path.is_file() or path.is_symlink():
                findings.append(Finding('MISSING_RESOURCE', 'error', self._rel(root, path), f'Required resource is missing for {content.content_id}.'))
            elif path.suffix == '.png':
                checks += 1
                self._validate_png(root, path, findings)
        if content.recipe:
            recipe_path = data / 'recipes' / f'{content.content_id}.json'
            checks += 1
            recipe = self._load_json(recipe_path, findings, root)
            result = recipe.get('result', {})
            if not isinstance(result, dict) or result.get('item') != f'{spec.mod_id}:{content.content_id}':
                findings.append(Finding('BAD_RECIPE_RESULT', 'error', self._rel(root, recipe_path), f'Recipe result does not target {content.content_id}.'))
        return checks

    def _validate_boss(self, root: Path, spec: ModSpec, en: dict[str, Any], ko: dict[str, Any], en_path: Path, ko_path: Path, findings: list[Finding]) -> int:
        boss = spec.boss
        assert boss is not None
        checks = 0
        package_path = Path(*spec.package_name.split('.'))
        main_class = _class_name(spec.mod_id) + 'Mod'
        entity_class = _class_name(boss.entity_id) + 'Entity'
        renderer_class = _class_name(boss.entity_id) + 'Renderer'
        required = (root / f'src/main/resources/assets/{spec.mod_id}/textures/entity/{boss.entity_id}.png', root / f'src/main/resources/assets/{spec.mod_id}/models/item/{boss.entity_id}_spawn_egg.json', root / f'src/main/resources/data/{spec.mod_id}/loot_tables/entities/{boss.entity_id}.json', root / 'src/main/java' / package_path / 'entity' / f'{entity_class}.java', root / 'src/main/java' / package_path / 'client' / f'{renderer_class}.java', root / 'src/main/java' / package_path / 'client' / f'{main_class}Client.java', root / f'.minecraft_ai/art_sources/{boss.entity_id}.bbmodel', root / f'.minecraft_ai/art_sources/{boss.entity_id}.obj', root / f'.minecraft_ai/art_sources/{boss.entity_id}.mtl')
        for path in required:
            checks += 1
            if not path.is_file() or path.is_symlink():
                findings.append(Finding('MISSING_BOSS_ASSET', 'error', self._rel(root, path), f'Boss asset is missing for {boss.entity_id}.'))
        checks += 1
        self._validate_png(root, required[0], findings)
        bbmodel = self._load_json(required[6], findings, root)
        checks += 1
        if bbmodel.get('model_identifier') != f'{spec.mod_id}:{boss.entity_id}':
            findings.append(Finding('BAD_BBMODEL', 'error', self._rel(root, required[6]), 'Blockbench model identifier does not match the boss.'))
        obj_text = required[7].read_text(encoding='utf-8', errors='replace') if required[7].is_file() else ''
        checks += 2
        if len(re.findall('^v\\s', obj_text, re.MULTILINE)) < 8:
            findings.append(Finding('BAD_OBJ', 'error', self._rel(root, required[7]), 'Boss OBJ has too few vertices.'))
        if len(re.findall('^f\\s', obj_text, re.MULTILINE)) < 6:
            findings.append(Finding('BAD_OBJ', 'error', self._rel(root, required[7]), 'Boss OBJ has too few faces.'))
        key = f'entity.{spec.mod_id}.{boss.entity_id}'
        for locale, translations, path in (('en_us', en, en_path), ('ko_kr', ko, ko_path)):
            checks += 1
            if key not in translations:
                findings.append(Finding('MISSING_BOSS_TRANSLATION', 'error', self._rel(root, path), f'{locale} is missing {key}.'))
        return checks

    def _validate_fabric_metadata(self, root: Path, spec: ModSpec, path: Path, fabric: dict[str, Any], complete: CompleteProposal | None, findings: list[Finding]) -> int:
        checks = 0
        expected = {'id': spec.mod_id, 'version': '${version}', 'environment': '*'}
        for key, value in expected.items():
            checks += 1
            if fabric.get(key) != value:
                findings.append(Finding('BAD_FABRIC_METADATA', 'error', self._rel(root, path), f'{key} must equal {value!r}.'))
        depends = fabric.get('depends')
        expected_depends = fabric_dependency_predicates(spec.platform)
        checks += 1
        if not isinstance(depends, dict):
            findings.append(Finding('BAD_FABRIC_DEPENDS', 'error', self._rel(root, path), 'depends must be an object.'))
        else:
            for dependency, constraint in expected_depends.items():
                checks += 1
                if depends.get(dependency) != constraint:
                    findings.append(Finding('BAD_FABRIC_DEPENDS', 'error', self._rel(root, path), f'{dependency} must equal {constraint!r}.'))
        entrypoints = fabric.get('entrypoints')
        checks += 1
        if not isinstance(entrypoints, dict):
            findings.append(Finding('BAD_ENTRYPOINTS', 'error', self._rel(root, path), 'entrypoints must be an object.'))
            return checks
        main_class = f'{spec.package_name}.{_class_name(spec.mod_id)}Mod'
        gametest_class = main_class + 'GameTests'
        for group, required in (('main', main_class), ('fabric-gametest', gametest_class)):
            checks += 1
            values = _entrypoint_values(entrypoints.get(group))
            if required not in values:
                findings.append(Finding('BAD_ENTRYPOINTS', 'error', self._rel(root, path), f'{group} must include {required}.'))
        if spec.boss is not None:
            required_client = f'{spec.package_name}.client.{_class_name(spec.mod_id)}ModClient'
            checks += 1
            if required_client not in _entrypoint_values(entrypoints.get('client')):
                findings.append(Finding('BAD_ENTRYPOINTS', 'error', self._rel(root, path), f'client must include {required_client}.'))
        if complete is not None and _complete_client_required(complete):
            checks += 1
            if not _entrypoint_values(entrypoints.get('client')):
                findings.append(Finding('COMPLETE_CLIENT_ENTRYPOINT_MISSING', 'error', self._rel(root, path), 'Approved complete project requires a client entrypoint.'))
        return checks

    def _validate_no_unapproved_client_or_entity(self, root: Path, spec: ModSpec, fabric: dict[str, Any], findings: list[Finding]) -> int:
        checks = 0
        entrypoints = fabric.get('entrypoints')
        checks += 1
        if isinstance(entrypoints, dict) and _entrypoint_values(entrypoints.get('client')):
            findings.append(Finding('UNAPPROVED_CLIENT_ARTIFACT', 'error', 'src/main/resources/fabric.mod.json', 'Project declares an unapproved client entrypoint.'))
        forbidden_roots = (root / f"src/main/java/{spec.package_name.replace('.', '/')}/entity", root / f'src/main/resources/assets/{spec.mod_id}/textures/entity', root / f'src/main/resources/data/{spec.mod_id}/loot_tables/entities')
        for forbidden_root in forbidden_roots:
            checks += 1
            if forbidden_root.is_dir() and any((path.is_file() for path in forbidden_root.rglob('*'))):
                findings.append(Finding('UNAPPROVED_ENTITY_ARTIFACT', 'error', self._rel(root, forbidden_root), 'Entity artifacts exist without an approved entity module.'))
        return checks

    def _validate_complete_sources(self, root: Path, spec: ModSpec, complete: CompleteProposal, findings: list[Finding]) -> int:
        checks = 0
        package_root = root / 'src/main/java' / Path(*spec.package_name.split('.'))
        module_kinds = {module.kind for module in complete.modules}
        extended_kinds = {'item', 'block', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine', 'effect', 'enchantment', 'command', 'recipe', 'advancement', 'loot'}
        if module_kinds & extended_kinds:
            checks += 1
            path = package_root / 'extended/GeneratedExtendedContent.java'
            if not path.is_file():
                findings.append(Finding('COMPLETE_EXTENDED_SOURCE_MISSING', 'error', self._rel(root, path), 'Approved extended content registrar is missing.'))
        system_classes = {'quest': 'QuestSystem', 'class': 'ClassSkillSystem', 'skill': 'ClassSkillSystem', 'economy': 'EconomyShopSystem', 'shop': 'EconomyShopSystem', 'gui': 'GuiNetworkingSystem', 'networking': 'GuiNetworkingSystem', 'party': 'PartyGuildSystem', 'guild': 'PartyGuildSystem'}
        for class_name in sorted({system_classes[kind] for kind in module_kinds if kind in system_classes}):
            checks += 1
            path = package_root / 'system' / f'{class_name}.java'
            if not path.is_file():
                findings.append(Finding('COMPLETE_SYSTEM_SOURCE_MISSING', 'error', self._rel(root, path), f'Approved system source is missing: {class_name}'))
        for entity_id in sorted(_complete_entity_ids(complete)):
            checks += 1
            path = package_root / 'entity' / f'{_class_name(entity_id)}Entity.java'
            if not path.is_file():
                findings.append(Finding('COMPLETE_ENTITY_SOURCE_MISSING', 'error', self._rel(root, path), f'Approved entity source is missing: {entity_id}'))
        if _complete_entity_ids(complete):
            for relative in ('geckolib/GeneratedGeckoEntities.java', 'client/geckolib/GeneratedGeckoClient.java'):
                checks += 1
                path = package_root / relative
                if not path.is_file():
                    findings.append(Finding('COMPLETE_GECKOLIB_SOURCE_MISSING', 'error', self._rel(root, path), 'Approved GeckoLib registrar is missing.'))
        checks += self._validate_local_ai_sidecar_sources(root, spec, complete, findings)
        return checks

    def _validate_local_ai_sidecar_sources(self, root: Path, spec: ModSpec, complete: CompleteProposal, findings: list[Finding]) -> int:
        checks = 0
        for module in complete.modules:
            if module.kind != 'integration' or module.config.get('integration_type') != LOCAL_AI_SIDECAR_INTEGRATION_TYPE:
                continue
            checks += 1
            try:
                normalized = normalize_local_ai_sidecar_config(module.config)
                source_relative = local_ai_sidecar_source_path(spec.package_name, module.module_id)
                manifest_relative = local_ai_sidecar_manifest_path(module.module_id)
                expected_source = render_local_ai_sidecar_source(package_name=spec.package_name, module_id=module.module_id, config=normalized)
                expected_manifest = render_local_ai_sidecar_manifest(package_name=spec.package_name, module_id=module.module_id, config=normalized)
            except LocalAiSidecarGenerationError as exc:
                findings.append(Finding('LOCAL_AI_SIDECAR_POLICY_INVALID', 'error', '.minecraft_ai/complete-proposal.json', f'Approved local AI sidecar policy is invalid: {exc}'))
                continue
            for relative, expected, code, label in ((source_relative, expected_source, 'LOCAL_AI_SIDECAR_SOURCE_MISMATCH', 'source'), (manifest_relative, expected_manifest, 'LOCAL_AI_SIDECAR_MANIFEST_MISMATCH', 'policy manifest')):
                checks += 1
                path = root / relative
                if not path.is_file() or path.is_symlink() or path.read_text(encoding='utf-8', errors='replace') != expected:
                    findings.append(Finding(code, 'error', relative, f'Reviewed local AI sidecar {label} must exactly match the approved deterministic reconstruction.'))
        return checks

    def _load_json(self, path: Path, findings: list[Finding], root: Path) -> dict[str, Any]:
        if not path.is_file() or path.is_symlink():
            findings.append(Finding('MISSING_JSON', 'error', self._rel(root, path), 'Required JSON is missing.'))
            return {}
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            findings.append(Finding('INVALID_JSON', 'error', self._rel(root, path), str(exc)))
            return {}
        if not isinstance(value, dict):
            findings.append(Finding('JSON_NOT_OBJECT', 'error', self._rel(root, path), 'Expected a JSON object.'))
            return {}
        return value

    def _validate_boss(
        self,
        root: Path,
        spec: Any,
        en: dict[str, Any],
        ko: dict[str, Any],
        en_path: Path,
        ko_path: Path,
        findings: list[Any],
    ) -> int:
        boss = spec.boss
        assert boss is not None
        checks = 0
        package_path = Path(*spec.package_name.split("."))
        main_class = _class_name(spec.mod_id) + "Mod"
        entity_class = _class_name(boss.entity_id) + "ModEntity"
        renderer_class = _class_name(boss.entity_id) + "ModRenderer"
        required = (
            root
            / f"src/main/resources/assets/{spec.mod_id}/textures/entity/{boss.entity_id}.png",
            root
            / f"src/main/resources/assets/{spec.mod_id}/models/item/{boss.entity_id}_spawn_egg.json",
            root
            / f"src/main/resources/data/{spec.mod_id}/loot_tables/entities/{boss.entity_id}.json",
            root / "src/main/java" / package_path / "entity" / f"{entity_class}.java",
            root / "src/main/java" / package_path / "client" / f"{renderer_class}.java",
            root / "src/main/java" / package_path / "client" / f"{main_class}Client.java",
            root / f".minecraft_ai/art_sources/{boss.entity_id}.bbmodel",
            root / f".minecraft_ai/art_sources/{boss.entity_id}.obj",
            root / f".minecraft_ai/art_sources/{boss.entity_id}.mtl",
        )
        for path in required:
            checks += 1
            if not path.is_file() or path.is_symlink():
                findings.append(
                    Finding(
                        "MISSING_BOSS_ASSET",
                        "error",
                        self._rel(root, path),
                        f"Boss asset is missing for {boss.entity_id}.",
                    )
                )
        checks += 1
        self._validate_png(root, required[0], findings)
        bbmodel = self._load_json(required[6], findings, root)
        checks += 1
        if bbmodel.get("model_identifier") != f"{spec.mod_id}:{boss.entity_id}":
            findings.append(
                Finding(
                    "BAD_BBMODEL",
                    "error",
                    self._rel(root, required[6]),
                    "Blockbench model identifier does not match the boss.",
                )
            )
        obj_text = (
            required[7].read_text(encoding="utf-8", errors="replace")
            if required[7].is_file()
            else ""
        )
        checks += 2
        if len(re.findall(r"^v\s", obj_text, re.MULTILINE)) < 8:
            findings.append(
                Finding(
                    "BAD_OBJ",
                    "error",
                    self._rel(root, required[7]),
                    "Boss OBJ has too few vertices.",
                )
            )
        if len(re.findall(r"^f\s", obj_text, re.MULTILINE)) < 6:
            findings.append(
                Finding(
                    "BAD_OBJ",
                    "error",
                    self._rel(root, required[7]),
                    "Boss OBJ has too few faces.",
                )
            )
        key = f"entity.{spec.mod_id}.{boss.entity_id}"
        for locale, translations, path in (
            ("en_us", en, en_path),
            ("ko_kr", ko, ko_path),
        ):
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
        return checks

    def _validate_png(self, root: Path, path: Path, findings: list[Finding]) -> None:
        if not path.is_file() or path.is_symlink():
            return
        with path.open('rb') as handle:
            data = handle.read(24)
        relative = self._rel(root, path)
        if len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n':
            findings.append(Finding('INVALID_PNG', 'error', relative, 'Invalid PNG signature.'))
            return
        width, height = struct.unpack('>II', data[16:24])
        if width < 1 or height < 1 or width > self.policy.max_texture_dimension or (height > self.policy.max_texture_dimension):
            findings.append(Finding('BAD_TEXTURE_SIZE', 'error', relative, 'Texture dimensions exceed MMM_MAX_TEXTURE_DIMENSION host policy.'))

    @staticmethod
    def _rel(root: Path, path: Path) -> str:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            return str(path)

def _load_complete_project_proposal(root: Path, spec: ModSpec, findings: list[Finding]) -> CompleteProposal | None:
    path = root / '.minecraft_ai/complete-proposal.json'
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
        if raw.get('schema_version') in {'mmm/complete-proposal-index-v1', 'mmm/complete-proposal-index-v2', 'mmm/complete-proposal-index-v3'}:
            from .proposal_store import load_sharded_complete_proposal
            proposal = load_sharded_complete_proposal(path)
        else:
            proposal = CompleteProposal.from_dict(raw)
        if proposal.status.value != 'approved':
            raise ValueError('complete proposal is not approved')
        if proposal.base_proposal.spec.mod_id != spec.mod_id:
            raise ValueError('complete proposal mod id mismatch')
        return proposal
    except Exception as exc:
        findings.append(Finding('INVALID_COMPLETE_PROPOSAL', 'error', path.relative_to(root).as_posix(), str(exc)))
        return None

def _reviewed_local_ai_sidecar_network_sources(root: Path, spec: ModSpec, complete: CompleteProposal | None) -> dict[str, str]:
    """Return only source whose approved policy manifest also matches exactly."""
    if complete is None:
        return {}
    reviewed: dict[str, str] = {}
    for module in complete.modules:
        if module.kind != 'integration' or module.config.get('integration_type') != LOCAL_AI_SIDECAR_INTEGRATION_TYPE:
            continue
        try:
            normalized = normalize_local_ai_sidecar_config(module.config)
            source_relative = local_ai_sidecar_source_path(spec.package_name, module.module_id)
            manifest_relative = local_ai_sidecar_manifest_path(module.module_id)
            expected_source = render_local_ai_sidecar_source(package_name=spec.package_name, module_id=module.module_id, config=normalized)
            expected_manifest = render_local_ai_sidecar_manifest(package_name=spec.package_name, module_id=module.module_id, config=normalized)
            source_path = root / source_relative
            manifest_path = root / manifest_relative
            if source_path.is_file() and (not source_path.is_symlink()) and manifest_path.is_file() and (not manifest_path.is_symlink()) and (source_path.read_text(encoding='utf-8') == expected_source) and (manifest_path.read_text(encoding='utf-8') == expected_manifest):
                reviewed[source_relative] = expected_source
        except (LocalAiSidecarGenerationError, OSError, UnicodeError):
            continue
    return reviewed

def _load_complete_jar_proposal(archive: zipfile.ZipFile, spec: ModSpec, findings: list[Finding]) -> CompleteProposal | None:
    name = 'META-INF/mmm-complete-proposal.json'
    if name not in archive.namelist():
        return None
    try:
        raw = json.loads(archive.read(name).decode('utf-8'))
        if raw.get('schema_version') in {'mmm/complete-proposal-index-v1', 'mmm/complete-proposal-index-v2', 'mmm/complete-proposal-index-v3'}:
            from .proposal_store import load_sharded_complete_proposal_from_zip
            proposal = load_sharded_complete_proposal_from_zip(archive, name)
        else:
            proposal = CompleteProposal.from_dict(raw)
        if proposal.status.value != 'approved':
            raise ValueError('complete proposal is not approved')
        if proposal.base_proposal.spec.mod_id != spec.mod_id:
            raise ValueError('complete proposal mod id mismatch')
        return proposal
    except Exception as exc:
        findings.append(Finding('JAR_INVALID_COMPLETE_PROPOSAL', 'error', name, str(exc)))
        return None

def _complete_entity_ids(proposal: CompleteProposal | None) -> set[str]:
    if proposal is None:
        return set()
    return {module.module_id for module in proposal.modules if module.kind in {'entity', 'boss', 'npc'} and module.config.get('implementation') != 'custom'}

def _complete_client_required(proposal: CompleteProposal | None) -> bool:
    if proposal is None:
        return False
    return bool(_complete_entity_ids(proposal)) or any((module.config.get('client_required') is True for module in proposal.modules))

def validate_jar(jar_path: Path, spec: ModSpec, *, policy: ScalePolicy | None=None) -> ValidationReport:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    findings: list[Finding] = []
    checks = 0
    jar_path = jar_path.expanduser().resolve()
    if not jar_path.is_file() or jar_path.is_symlink():
        return ValidationReport(status='FAIL', checks_run=1, findings=(Finding('JAR_MISSING', 'error', str(jar_path), 'Built JAR is missing or unsafe.'),))
    checks += 1
    if not zipfile.is_zipfile(jar_path):
        return ValidationReport(status='FAIL', checks_run=checks + 1, findings=(Finding('NOT_A_JAR', 'error', str(jar_path), 'File is not a ZIP/JAR archive.'),))
    with zipfile.ZipFile(jar_path) as archive:
        infos = archive.infolist()
        raw_names = [item.filename for item in infos]
        names = set(raw_names)
        complete = _load_complete_jar_proposal(archive, spec, findings)
        checks += 1
        if len(names) != len(raw_names):
            findings.append(Finding('JAR_DUPLICATE_ENTRY', 'error', str(jar_path), 'JAR has duplicate entries.'))
        for info in infos:
            checks += 1
            normalized = info.filename.replace('\\', '/')
            if normalized.startswith('/') or '..' in Path(normalized).parts:
                findings.append(Finding('JAR_UNSAFE_PATH', 'error', info.filename, 'Unsafe path found in JAR.'))
            if info.file_size > policy.max_single_file_bytes:
                findings.append(Finding('JAR_ENTRY_TOO_LARGE', 'error', info.filename, 'JAR entry exceeds MMM_MAX_SINGLE_FILE_BYTES host policy.'))
        checks += 1
        corrupt = archive.testzip()
        if corrupt is not None:
            findings.append(Finding('JAR_BAD_CRC', 'error', corrupt, 'JAR entry failed its CRC check.'))
        metadata: dict[str, Any] = {}
        if 'fabric.mod.json' not in names:
            findings.append(Finding('JAR_NO_METADATA', 'error', str(jar_path), 'fabric.mod.json is absent.'))
        else:
            checks += 1
            try:
                metadata = json.loads(archive.read('fabric.mod.json').decode('utf-8'))
                if not isinstance(metadata, dict):
                    raise ValueError('fabric.mod.json is not an object')
            except Exception as exc:
                findings.append(Finding('JAR_BAD_METADATA', 'error', 'fabric.mod.json', str(exc)))
                metadata = {}
        expected_metadata = {'id': spec.mod_id, 'version': spec.version, 'environment': '*'}
        for key, value in expected_metadata.items():
            checks += 1
            if metadata.get(key) != value:
                findings.append(Finding('JAR_BAD_METADATA', 'error', 'fabric.mod.json', f'{key} must equal {value!r}.'))
        depends = metadata.get('depends')
        expected_depends = fabric_dependency_predicates(spec.platform)
        checks += 1
        if not isinstance(depends, dict):
            findings.append(Finding('JAR_BAD_DEPENDS', 'error', 'fabric.mod.json', 'depends must be an object.'))
        else:
            for dependency, constraint in expected_depends.items():
                checks += 1
                if depends.get(dependency) != constraint:
                    findings.append(Finding('JAR_BAD_DEPENDS', 'error', 'fabric.mod.json', f'{dependency} must equal {constraint!r}.'))
        class_names = {name for name in names if name.endswith('.class')}
        checks += 1
        if not class_names:
            findings.append(Finding('JAR_NO_CLASSES', 'error', str(jar_path), 'No class files found.'))
        entrypoints = metadata.get('entrypoints')
        checks += 1
        if not isinstance(entrypoints, dict):
            findings.append(Finding('JAR_BAD_ENTRYPOINTS', 'error', 'fabric.mod.json', 'entrypoints must be an object.'))
            entrypoints = {}
        for group, raw in entrypoints.items():
            for value in _entrypoint_values(raw):
                checks += 1
                class_path = value.replace('.', '/') + '.class'
                if class_path not in names:
                    findings.append(Finding('JAR_CLASS_MISSING', 'error', class_path, f'Declared {group} entrypoint class is absent.'))
                elif not archive.read(class_path).startswith(b'\xca\xfe\xba\xbe'):
                    findings.append(Finding('JAR_BAD_CLASS', 'error', class_path, 'Class entry does not have JVM class-file magic.'))
        required_entrypoints = {'main': f'{spec.package_name}.{_class_name(spec.mod_id)}Mod', 'fabric-gametest': f'{spec.package_name}.{_class_name(spec.mod_id)}ModGameTests'}
        if spec.boss is not None:
            required_entrypoints['client'] = f'{spec.package_name}.client.{_class_name(spec.mod_id)}ModClient'
        for group, value in required_entrypoints.items():
            checks += 1
            if value not in _entrypoint_values(entrypoints.get(group)):
                findings.append(Finding('JAR_BAD_ENTRYPOINTS', 'error', 'fabric.mod.json', f'{group} must include {value}.'))
        required_resources = {f'assets/{spec.mod_id}/lang/en_us.json', f'assets/{spec.mod_id}/lang/ko_kr.json'}
        for content in spec.contents:
            if content.kind is ContentKind.ITEM:
                required_resources.update({f'assets/{spec.mod_id}/models/item/{content.content_id}.json', f'assets/{spec.mod_id}/textures/item/{content.content_id}.png'})
            else:
                required_resources.update({f'assets/{spec.mod_id}/blockstates/{content.content_id}.json', f'assets/{spec.mod_id}/models/block/{content.content_id}.json', f'assets/{spec.mod_id}/models/item/{content.content_id}.json', f'assets/{spec.mod_id}/textures/block/{content.content_id}.png', f'data/{spec.mod_id}/loot_tables/blocks/{content.content_id}.json'})
            if content.recipe:
                required_resources.add(f'data/{spec.mod_id}/recipes/{content.content_id}.json')
        if spec.boss is not None:
            required_resources.update({f'assets/{spec.mod_id}/textures/entity/{spec.boss.entity_id}.png', f'assets/{spec.mod_id}/models/item/{spec.boss.entity_id}_spawn_egg.json', f'data/{spec.mod_id}/loot_tables/entities/{spec.boss.entity_id}.json'})
        for required in required_resources:
            checks += 1
            if required not in names:
                findings.append(Finding('JAR_RESOURCE_MISSING', 'error', required, 'Approved runtime resource is absent from JAR.'))
        if complete is not None:
            checks += _validate_complete_jar(archive, names, spec, complete, findings)
    status = 'PASS' if not any((item.severity == 'error' for item in findings)) else 'FAIL'
    return ValidationReport(status=status, checks_run=checks, findings=tuple(findings))

def _validate_complete_jar(archive: zipfile.ZipFile, names: set[str], spec: ModSpec, complete: CompleteProposal, findings: list[Finding]) -> int:
    checks = 0
    java_root = spec.package_name.replace('.', '/')
    module_kinds = {module.kind for module in complete.modules}
    extended_kinds = {'item', 'block', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine', 'effect', 'enchantment', 'command', 'recipe', 'advancement', 'loot'}
    expected_classes: set[str] = set()
    if module_kinds & extended_kinds:
        expected_classes.add(f'{java_root}/extended/GeneratedExtendedContent.class')
    system_classes = {'quest': 'QuestSystem', 'class': 'ClassSkillSystem', 'skill': 'ClassSkillSystem', 'economy': 'EconomyShopSystem', 'shop': 'EconomyShopSystem', 'gui': 'GuiNetworkingSystem', 'networking': 'GuiNetworkingSystem', 'party': 'PartyGuildSystem', 'guild': 'PartyGuildSystem'}
    expected_classes.update((f'{java_root}/system/{system_classes[kind]}.class' for kind in module_kinds if kind in system_classes))
    if module_kinds & set(system_classes):
        expected_classes.update({f'{java_root}/system/MmmPersistentStore.class', f'{java_root}/system/MmmSystemConfig.class'})
    for entity_id in _complete_entity_ids(complete):
        entity_class = _class_name(entity_id)
        expected_classes.update({f'{java_root}/entity/{entity_class}Entity.class', f'{java_root}/client/geckolib/{entity_class}GeoModel.class', f'{java_root}/client/geckolib/{entity_class}GeoRenderer.class'})
    if _complete_entity_ids(complete):
        expected_classes.update({f'{java_root}/geckolib/GeneratedGeckoEntities.class', f'{java_root}/client/geckolib/GeneratedGeckoClient.class'})
    for class_name in expected_classes:
        checks += 1
        if class_name not in names:
            findings.append(Finding('JAR_CLASS_MISSING', 'error', class_name, 'Approved complete support class is absent.'))
    return checks

def _entrypoint_values(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    result: set[str] = set()
    for item in value:
        if isinstance(item, str):
            result.add(item)
        elif isinstance(item, dict) and isinstance(item.get('value'), str):
            result.add(item['value'])
    return result

def _class_name(value: str) -> str:
    return ''.join((part.capitalize() for part in value.split('_')))
