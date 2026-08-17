import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from minecraft_mod_ai.complete_orchestrator import CompleteExecutionOptions, CompleteProductionError, CompleteProductionOrchestrator
from minecraft_mod_ai.complete_spec import CompleteProposal, CompleteProposalStatus, ProductionModule, complete_proposal_from_parts
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.geckolib_generator import generate_geckolib_entity_assets
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.project_edit import inspect_fabric_project
from minecraft_mod_ai.source_patch import SourcePatchError, TransactionalSourcePatcher, sha256_file
from minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec, SpecValidationError

def _spec() -> ModSpec:
    return ModSpec(mod_id='complete_test', mod_name='Complete Test', package_name='ai.minecraft.complete_test', version='1.0.0', summary='complete production test', contents=(ContentSpec(content_id='core_item', kind=ContentKind.ITEM, display_name_en='Core Item', display_name_ko='핵심 아이템'),))

def _project(root: Path) -> Path:
    FabricProjectGenerator().generate(_spec(), root)
    return root

def test_complete_proposal_hash_covers_modules_and_rejects_cycles() -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    proposal = complete_proposal_from_parts(requested_prompt='Create quests and a weapon', base_proposal=base, game_design={'title': 'Complete'}, modules=(ProductionModule('weapon_one', 'weapon', {'attack_damage': 5}), ProductionModule('quest_one', 'quest', {}, ('weapon_one',))), acceptance_tests=('weapon and quest work',))
    assert proposal.approve(proposal.calculate_hash()).status is CompleteProposalStatus.APPROVED
    raw = proposal.to_dict()
    raw['modules'][0]['config']['attack_damage'] = 99
    with pytest.raises(SpecValidationError):
        CompleteProposal.from_dict(raw)
    with pytest.raises(SpecValidationError):
        complete_proposal_from_parts(requested_prompt='cycle', base_proposal=base, game_design={'title': 'Cycle'}, modules=(ProductionModule('left', 'custom_java', {}, ('right',)), ProductionModule('right', 'custom_java', {}, ('left',))), acceptance_tests=('never',))

def test_source_patch_is_hash_guarded_and_transactional(tmp_path: Path) -> None:
    root = tmp_path / 'project'
    root.mkdir()
    target = root / 'file.txt'
    target.write_text('before\n', encoding='utf-8')
    patcher = TransactionalSourcePatcher(root)
    receipt = patcher.apply([{'operation': 'edit', 'path': 'file.txt', 'expected_sha256': sha256_file(target), 'replacements': [{'old': 'before', 'new': 'after', 'count': 1}]}, {'operation': 'create', 'path': 'new.txt', 'content': 'new\n'}])
    assert receipt['status'] == 'APPLIED'
    assert target.read_text(encoding='utf-8') == 'after\n'
    with pytest.raises(SourcePatchError):
        patcher.apply([{'operation': 'replace', 'path': 'file.txt', 'expected_sha256': 'sha256:' + '0' * 64, 'content': 'corrupt'}])
    assert target.read_text(encoding='utf-8') == 'after\n'

def test_geckolib_generator_accumulates_real_entity_bindings(tmp_path: Path) -> None:
    project = _project(tmp_path / 'project')
    for entity_id in ('frost_guard', 'ember_guard'):
        result = generate_geckolib_entity_assets(project_root=project, mod_id='complete_test', package_name='ai.minecraft.complete_test', entity_id=entity_id)
        assert result['status'] == 'fabric_binding_generated'
    registrar = project / 'src/main/java/ai/minecraft/complete_test/geckolib/GeneratedGeckoEntities.java'
    text = registrar.read_text(encoding='utf-8')
    assert 'forEachDescriptor' in text
    assert 'GeckoLib.initialize()' in text
    assert 'FROST_GUARD' not in text and 'EMBER_GUARD' not in text
    server_units = sorted(project.rglob('*GeckoRegistration.java'))
    assert len(server_units) == 2
    unit_text = '\n'.join((path.read_text(encoding='utf-8') for path in server_units))
    assert 'FROST_GUARD' in unit_text and 'EMBER_GUARD' in unit_text
    assert 'FabricDefaultAttributeRegistry.register' in unit_text
    entity_java = project / 'src/main/java/ai/minecraft/complete_test/entity/FrostGuardEntity.java'
    entity_text = entity_java.read_text(encoding='utf-8')
    assert 'animation.complete_test.frost_guard.idle' in entity_text
    assert 'animation.complete_test.frost_guard.attack' in entity_text
    client = project / 'src/main/java/ai/minecraft/complete_test/client/geckolib/GeneratedGeckoClient.java'
    assert 'forEachDescriptor' in client.read_text(encoding='utf-8')
    client_units = sorted(project.rglob('*GeckoClientRegistration.java'))
    assert len(client_units) == 2
    assert sum((path.read_text(encoding='utf-8').count('EntityRendererRegistry.register') for path in client_units)) == 2

def test_complete_orchestrator_source_only_connects_all_generators(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    proposal = complete_proposal_from_parts(requested_prompt='weapon, quest, animated entity and menu', base_proposal=base, game_design={'title': 'Integrated'}, modules=(ProductionModule('frost_blade', 'weapon', {'attack_damage': 6}), ProductionModule('first_quest', 'quest', {}, ('frost_blade',)), ProductionModule('frost_guard', 'entity', {'max_health': 60}), ProductionModule('status_menu', 'gui', {'template': 'read_only_menu'})), acceptance_tests=('all generated systems are present',))
    result = CompleteProductionOrchestrator(workspace_root=tmp_path / 'out').execute(proposal, approval_hash=proposal.calculate_hash(), run_name='integrated', options=CompleteExecutionOptions(source_only=True, run_jdt=False, run_blockbench=False, run_runtime=False, run_client=False, run_mineflayer=False, run_visual_review=False))
    assert result.status == 'SOURCE_READY'
    project = Path(result.project_root)
    package_path = Path(*base.spec.package_name.split('.'))
    assert (project / 'src/main/java' / package_path / 'extended/GeneratedExtendedContent.java').is_file()
    assert any((path.name == 'QuestSystem.java' for path in project.rglob('QuestSystem.java')))
    assert any((path.name == 'GeneratedGeckoEntities.java' for path in project.rglob('GeneratedGeckoEntities.java')))
    assert not list(project.rglob('GeneratedWorldRuntime.java'))
    with zipfile.ZipFile(result.release_zip) as archive:
        assert any((name.endswith('GeneratedExtendedContent.java') for name in archive.namelist()))

def test_v2_source_only_persists_fail_closed_quality_convergence(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    modules = (ProductionModule('frost_item', 'item'),)
    game_design = {'title': 'Evidence-bound source build'}
    compiled = compile_production_contract(requested_prompt='Create one frost item', game_design=game_design, modules=modules, acceptance_tests=('the requested item exists',))
    proposal = complete_proposal_from_parts(requested_prompt='Create one frost item', base_proposal=base, game_design={**game_design, '_production_contract': compiled.contract}, modules=modules, acceptance_tests=compiled.acceptance_tests)
    result = CompleteProductionOrchestrator(workspace_root=tmp_path / 'out').execute(proposal, approval_hash=proposal.calculate_hash(), run_name='quality-source', options=CompleteExecutionOptions(source_only=True, run_jdt=False, run_blockbench=False, run_runtime=False, run_client=False, run_mineflayer=False, run_visual_review=False))
    assert result.status == 'SOURCE_READY'
    assert result.release_ready is False
    assert result.quality_report is not None
    assert result.quality_report['overall_status'] == 'MISSING'
    assert 'quality:runtime' in result.unresolved_gates
    run_root = Path(result.work_ledger_path).parent.parent
    persisted = run_root / '.minecraft_ai/quality-convergence.json'
    assert persisted.is_file()
    assert json.loads(persisted.read_text(encoding='utf-8')) == result.quality_report

def test_complete_orchestrator_requires_exact_existing_input_presence(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    unbound = complete_proposal_from_parts(requested_prompt='new project', base_proposal=base, game_design={'title': 'Fresh'}, modules=(ProductionModule('fresh_item', 'item'),), acceptance_tests=('item exists',))
    archive = tmp_path / 'existing.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('project/readme.txt', 'input')
    archive_hash = 'sha256:' + hashlib.sha256(archive.read_bytes()).hexdigest()
    bound = complete_proposal_from_parts(requested_prompt='revise project', base_proposal=base, game_design={'title': 'Revision'}, modules=(ProductionModule('revision_item', 'item'),), acceptance_tests=('item exists',), existing_input_sha256=archive_hash)
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / 'runs')
    with pytest.raises(CompleteProductionError, match='same ZIP is required'):
        orchestrator.execute(bound, approval_hash=bound.calculate_hash(), run_name='missing-input', options=CompleteExecutionOptions(source_only=True))
    with pytest.raises(CompleteProductionError, match='approved with that input'):
        orchestrator.execute(unbound, approval_hash=unbound.calculate_hash(), run_name='unexpected-input', options=CompleteExecutionOptions(source_only=True), existing_input=archive)
    assert list((tmp_path / 'runs').iterdir()) == []

def test_complete_orchestrator_checks_bound_hash_before_extraction(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    archive = tmp_path / 'existing.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('project/readme.txt', 'approved bytes')
    archive_hash = 'sha256:' + hashlib.sha256(archive.read_bytes()).hexdigest()
    proposal = complete_proposal_from_parts(requested_prompt='revise project', base_proposal=base, game_design={'title': 'Revision'}, modules=(ProductionModule('revision_item', 'item'),), acceptance_tests=('item exists',), existing_input_sha256=archive_hash)
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('project/readme.txt', 'changed bytes')
    runs = tmp_path / 'runs'
    with pytest.raises(CompleteProductionError, match='changed after complete-plan approval'):
        CompleteProductionOrchestrator(workspace_root=runs).execute(proposal, approval_hash=proposal.calculate_hash(), run_name='changed-input', options=CompleteExecutionOptions(source_only=True), existing_input=archive)
    assert not (runs / 'changed-input/existing-source').exists()

def test_prepare_project_archives_partial_deterministic_directory(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    awaiting = complete_proposal_from_parts(requested_prompt='new project', base_proposal=base, game_design={'title': 'Fresh'}, modules=(ProductionModule('fresh_item', 'item'),), acceptance_tests=('item exists',))
    proposal = awaiting.approve(awaiting.calculate_hash())
    run_root = tmp_path / 'run'
    partial = run_root / 'base/workspaces' / base.spec.mod_id
    partial.mkdir(parents=True)
    (partial / 'partial.txt').write_text('preserve me\n', encoding='utf-8')
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / 'workspace')
    project = orchestrator._prepare_project(proposal, run_root=run_root, existing_input=None)
    assert orchestrator._valid_project_root(project)
    preserved = partial.with_name(partial.name + '.incomplete-1')
    assert (preserved / 'partial.txt').read_text(encoding='utf-8') == 'preserve me\n'

def test_required_gate_matrix_is_receipt_backed_and_fail_closed(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    proposal = complete_proposal_from_parts(requested_prompt='verified item', base_proposal=base, game_design={'title': 'Verified'}, modules=(ProductionModule('verified_item', 'item', required_gates=('JDT diagnostics', 'Gradle clean build', 'GameTest', 'JAR validation', 'runtime interaction tests', 'visual review')),), acceptance_tests=('item works',))
    main_class = ''.join((part.capitalize() for part in base.spec.mod_id.split('_'))) + 'Mod'
    gametest_report = tmp_path / 'gametest-report.xml'
    gametest_report.write_text(f'<testsuite failures="0" errors="0" skipped="0"><testcase name="{main_class}GameTests.generatedRegistriesAreLive"/></testsuite>', encoding='utf-8')
    build = {'status': 'PASS', 'gametest_report': str(gametest_report), 'commands': [{'name': 'clean_build', 'exit_code': 0, 'timed_out': False}, {'name': 'gametest', 'exit_code': 0, 'timed_out': False}]}
    common = {'source_validation': {'status': 'PASS'}, 'jdt_receipt': {'files_opened': 4, 'error_count': 0}, 'build_report': build, 'jar_validation': {'status': 'PASS'}, 'blockbench_receipts': (), 'runtime_receipt': None, 'playtest_receipt': {'status': 'PASS', 'interaction_count': 1, 'assertion_count': 1}, 'visual_receipt': {'status': 'PASS'}}
    assert CompleteProductionOrchestrator._required_gate_failures(proposal, generated_receipts=(), **common) == []
    failures = CompleteProductionOrchestrator._required_gate_failures(proposal, generated_receipts=({'required_gates': ['restart persistence test']},), **{**common, 'build_report': {'status': 'PASS', 'commands': [build['commands'][0]]}})
    assert any(('GameTest:missing-gametest' in item for item in failures))
    assert any(('restart persistence test:unsupported' in item for item in failures))

def test_prepare_project_recovers_one_nested_release_source_zip(tmp_path: Path) -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one frost item')
    spec = base.spec
    main_class = 'ImportedMod'
    main_entrypoint = f'{spec.package_name}.{main_class}'
    nested_path = tmp_path / 'editable-source.zip'
    with zipfile.ZipFile(nested_path, 'w') as nested:
        nested.writestr('build.gradle', "plugins { id 'fabric-loom' }\n")
        nested.writestr('src/main/resources/fabric.mod.json', json.dumps({'schemaVersion': 1, 'id': spec.mod_id, 'name': spec.mod_name, 'version': spec.version, 'entrypoints': {'main': [main_entrypoint]}, 'depends': {'minecraft': spec.platform.minecraft_version}}))
        nested.writestr('src/main/java/' + spec.package_name.replace('.', '/') + f'/{main_class}.java', f'package {spec.package_name}; public final class {main_class} {{}}\n')
    release = tmp_path / 'release.zip'
    with zipfile.ZipFile(release, 'w') as outer:
        outer.writestr(f'source/{spec.mod_id}-{spec.version}-source.zip', nested_path.read_bytes())
    archive_hash = 'sha256:' + hashlib.sha256(release.read_bytes()).hexdigest()
    awaiting = complete_proposal_from_parts(requested_prompt='revise nested project', base_proposal=base, game_design={'title': 'Nested Revision'}, modules=(ProductionModule('nested_item', 'item'),), acceptance_tests=('item exists',), existing_input_sha256=archive_hash)
    approved = awaiting.approve(awaiting.calculate_hash())
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / 'workspace')
    project = orchestrator._prepare_project(approved, run_root=tmp_path / 'run', existing_input=release)
    info = inspect_fabric_project(project)
    assert info.mod_id == spec.mod_id
    assert info.package_name == spec.package_name
    assert 'existing-source-nested' in project.as_posix()
