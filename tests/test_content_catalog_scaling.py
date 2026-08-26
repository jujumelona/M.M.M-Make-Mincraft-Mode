import json
from pathlib import Path

from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.extended_content_generator import (
    generate_extended_content,
    iter_extended_module_records,
)
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.scale_policy import ScalePolicy
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec
from minecraft_mod_ai.system_pack_generator import (
    generate_system_pack,
    iter_system_module_records,
)


def _project(root: Path) -> Path:
    spec = ModSpec(mod_id='catalog_test', mod_name='Catalog Test', package_name='ai.minecraft.catalog_test', version='1.0.0', summary='bounded catalog scaling test', contents=(ContentSpec(content_id='bootstrap_item', kind=ContentKind.ITEM, display_name_en='Bootstrap Item', display_name_ko='Bootstrap Item'),))
    FabricProjectGenerator().generate(spec, root)
    return root

def _extended_metrics(root: Path, count: int) -> tuple[int, int]:
    project = _project(root)
    modules = tuple(ProductionModule(module_id=f'command_{index:05d}', kind='command', config={'literal': f'catalog{index:05d}', 'message': f'Catalog command {index:05d}'}) for index in range(count))
    result = generate_extended_content(project_root=project, mod_id='catalog_test', package_name='ai.minecraft.catalog_test', modules=modules, policy=ScalePolicy(java_shard_size=8))
    catalog_root = project / '.minecraft_ai/extended-modules.json'
    catalog = json.loads(catalog_root.read_text(encoding='utf-8'))
    assert catalog['schema_version'] == 'mmm/extended-module-directory-v1'
    assert catalog['module_count'] == count
    assert 'modules' not in catalog
    assert len(list(iter_extended_module_records(project))) == count
    metadata_files = [catalog_root, *sorted((project / '.minecraft_ai/extended-module-records').rglob('*.json'))]
    java_root = project / 'src/main/java/ai/minecraft/catalog_test/extended'
    java_files = sorted(java_root.glob('GeneratedContent*.java'))
    registrar = java_root / 'GeneratedExtendedContent.java'
    registrar_text = registrar.read_text(encoding='utf-8')
    assert 'GeneratedContentUnit' in registrar_text
    assert 'Files.list(directory)' in registrar_text
    assert result['shard_count'] == count
    return (max(path.stat().st_size for path in metadata_files), max(path.stat().st_size for path in java_files))

def test_extended_catalog_and_registrar_files_stay_bounded_as_modules_grow(tmp_path: Path) -> None:
    small_metadata, small_java = _extended_metrics(tmp_path / 'small-extended', 64)
    large_metadata, large_java = _extended_metrics(tmp_path / 'large-extended', 2048)
    assert large_metadata <= small_metadata + 512
    assert large_java <= small_java + 512

def test_extended_catalog_reader_preserves_legacy_monolith_compatibility(tmp_path: Path) -> None:
    metadata = tmp_path / '.minecraft_ai'
    metadata.mkdir()
    (metadata / 'extended-modules.json').write_text(json.dumps({'schema_version': 'mmm/extended-modules-v2', 'shard_size': 8, 'modules': [{'module_id': 'legacy_item', 'kind': 'item', 'config': {}, 'depends_on': [], 'required_gates': []}]}), encoding='utf-8')
    assert [item['module_id'] for item in iter_extended_module_records(tmp_path)] == ['legacy_item']

def test_extended_generation_updates_only_the_new_content_unit(tmp_path: Path) -> None:
    project = _project(tmp_path / 'incremental')
    first = ProductionModule(module_id='first_command', kind='command', config={'literal': 'first', 'message': 'First'})
    second = ProductionModule(module_id='second_command', kind='command', config={'literal': 'second', 'message': 'Second'})
    generate_extended_content(project_root=project, mod_id='catalog_test', package_name='ai.minecraft.catalog_test', modules=(first,), policy=ScalePolicy(java_shard_size=8))
    units_before = sorted(project.rglob('GeneratedContentUnit*.java'))
    assert len(units_before) == 1
    first_bytes = units_before[0].read_bytes()
    receipt = generate_extended_content(project_root=project, mod_id='catalog_test', package_name='ai.minecraft.catalog_test', modules=(second,), policy=ScalePolicy(java_shard_size=8))
    units_after = sorted(project.rglob('GeneratedContentUnit*.java'))
    assert len(units_after) == 2
    assert units_before[0].read_bytes() == first_bytes
    assert str(units_before[0]) not in receipt['files']
    assert {item['module_id'] for item in iter_extended_module_records(project)} == {'first_command', 'second_command'}

def _system_modules(count: int) -> list[dict]:
    return [{'module_id': f'quest_{index:05d}', 'kind': 'quest', 'config': {'objective': 'manual', 'required': 1}, 'depends_on': [], 'required_gates': []} for index in range(count)]

def _system_contract_metrics(root: Path, count: int) -> int:
    project = _project(root)
    result = generate_system_pack(project_root=project, pack_id='quest-system', mod_id='catalog_test', package_name='ai.minecraft.catalog_test', config={'modules': _system_modules(count)}, policy=ScalePolicy(java_shard_size=8))
    resources = project / 'src/main/resources'
    contract = resources / 'data/catalog_test/mmm_systems/quest-system.json'
    root_index = json.loads(contract.read_text(encoding='utf-8'))
    assert root_index['storage_schema_version'] == 'mmm/system-pack-directory-v1'
    assert root_index['module_count'] == count
    assert 'modules' not in root_index
    assert 'root' not in root_index
    records = sorted((resources / root_index['directory'].removeprefix('/')).glob('*.json'))
    assert len(records) == count
    for path in records:
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['schema_version'] == 'mmm/system-module-record-v1'
        assert record['module']['module_id'] == path.stem
    assert len(iter_system_module_records(project, mod_id='catalog_test', pack_id='quest-system')) == count
    assert result['definition_shard_count'] == (count + 7) // 8
    assert result['definition_record_count'] == count
    loader = next(project.rglob('MmmSystemConfig.java')).read_text(encoding='utf-8')
    quest = next(project.rglob('QuestSystem.java')).read_text(encoding='utf-8')
    assert 'Files.list(directory)' in loader
    assert 'forEachModule' in quest
    assert 'getAsJsonArray("modules")' not in quest
    contract_files = [contract, *records]
    return max(path.stat().st_size for path in contract_files)

def test_system_contract_files_stay_bounded_as_modules_grow(tmp_path: Path) -> None:
    small = _system_contract_metrics(tmp_path / 'small-system', 64)
    large = _system_contract_metrics(tmp_path / 'large-system', 2048)
    assert large <= small + 512

def test_system_pack_incremental_shards_merge_and_update_one_record(tmp_path: Path) -> None:
    project = _project(tmp_path / 'incremental-system')
    modules = _system_modules(120)
    for shard in (modules[:48], modules[48:96], modules[96:]):
        result = generate_system_pack(project_root=project, pack_id='quest-system', mod_id='catalog_test', package_name='ai.minecraft.catalog_test', config={'modules': shard}, policy=ScalePolicy(java_shard_size=48))
    loaded = iter_system_module_records(project, mod_id='catalog_test', pack_id='quest-system')
    assert len(loaded) == 120
    assert {item['module_id'] for item in loaded} == {f'quest_{index:05d}' for index in range(120)}
    assert result['definition_count'] == 120
    assert result['definition_shard_count'] == 3
    resources = project / 'src/main/resources'
    contract = resources / 'data/catalog_test/mmm_systems/quest-system.json'
    records = resources / 'data/catalog_test/mmm_systems/quest-system/records'
    root_before = contract.read_bytes()
    loader_before = next(project.rglob('MmmSystemConfig.java')).read_bytes()
    system_before = next(project.rglob('QuestSystem.java')).read_bytes()
    unchanged_before = (records / 'quest_00000.json').read_bytes()
    replacement = {**modules[77], 'config': {'objective': 'manual', 'required': 9}}
    update = generate_system_pack(project_root=project, pack_id='quest-system', mod_id='catalog_test', package_name='ai.minecraft.catalog_test', config={'modules': [replacement]}, policy=ScalePolicy(java_shard_size=48))
    changed_paths = {item['path'] for item in update['write_receipt']['operations']}
    assert changed_paths == {'src/main/resources/data/catalog_test/mmm_systems/quest-system/records/quest_00077.json'}
    assert contract.read_bytes() == root_before
    assert next(project.rglob('MmmSystemConfig.java')).read_bytes() == loader_before
    assert next(project.rglob('QuestSystem.java')).read_bytes() == system_before
    assert (records / 'quest_00000.json').read_bytes() == unchanged_before
    updated = {item['module_id']: item for item in iter_system_module_records(project, mod_id='catalog_test', pack_id='quest-system')}
    assert updated['quest_00077']['config']['required'] == 9

def test_system_pack_migrates_legacy_shards_without_losing_modules(tmp_path: Path) -> None:
    project = _project(tmp_path / 'legacy-system')
    resources = project / 'src/main/resources'
    contract = resources / 'data/catalog_test/mmm_systems/quest-system.json'
    shard = resources / 'data/catalog_test/mmm_systems/quest-system/shards/modules-00000000.json'
    shard.parent.mkdir(parents=True)
    legacy_modules = _system_modules(2)
    shard.write_text(json.dumps({'schema_version': 'mmm/system-module-shard-v1', 'modules': legacy_modules}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_text(json.dumps({'schema_version': 'mmm/quest-system-v5', 'storage_schema_version': 'mmm/system-pack-index-v1', 'pack_id': 'quest-system', 'module_count': 2, 'shard_size': 8, 'root': '/data/catalog_test/mmm_systems/quest-system/shards/modules-00000000.json', 'server_authoritative': True, 'persistent': True, 'minecraft_version': '1.20.1', 'loader': 'fabric'}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    generate_system_pack(project_root=project, pack_id='quest-system', mod_id='catalog_test', package_name='ai.minecraft.catalog_test', config={'modules': [_system_modules(3)[2]]}, policy=ScalePolicy(java_shard_size=8))
    migrated = iter_system_module_records(project, mod_id='catalog_test', pack_id='quest-system')
    assert [item['module_id'] for item in migrated] == ['quest_00000', 'quest_00001', 'quest_00002']
    root = json.loads(contract.read_text(encoding='utf-8'))
    assert root['storage_schema_version'] == 'mmm/system-pack-directory-v1'

def test_class_skill_template_replays_bounded_catalog_for_dependency_passes(tmp_path: Path) -> None:
    project = _project(tmp_path / 'class-skill')
    generate_system_pack(project_root=project, pack_id='class-skill-system', mod_id='catalog_test', package_name='ai.minecraft.catalog_test', config={'modules': [{'module_id': 'ranger', 'kind': 'class', 'config': {'display_name': 'Ranger'}, 'depends_on': [], 'required_gates': []}, {'module_id': 'ranger_dash', 'kind': 'skill', 'config': {'required_class': 'ranger', 'effect': 'minecraft:speed'}, 'depends_on': ['ranger'], 'required_gates': []}]}, policy=ScalePolicy(java_shard_size=1))
    java = next(project.rglob('ClassSkillSystem.java')).read_text(encoding='utf-8')
    assert java.count('MmmSystemConfig.forEachModule') == 2
