from __future__ import annotations
import json
from pathlib import Path
from minecraft_mod_ai.complete_orchestrator import CompleteExecutionOptions, CompleteProductionOrchestrator
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.local_ai_sidecar_generator import generate_local_ai_sidecar, local_ai_sidecar_manifest_path, local_ai_sidecar_source_path
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.validator import ProjectValidator

def _base_proposal():
    return MinecraftModPipeline(planner=HeuristicPlanner()).plan('Create one local AI bridge anchor item')

def _generate_approved_project(root: Path) -> tuple[Path, object, ProductionModule]:
    module = _module()
    complete = _approved_complete(module)
    spec = complete.base_proposal.spec
    FabricProjectGenerator().generate(spec, root)
    generate_local_ai_sidecar(project_root=root, mod_id=spec.mod_id, package_name=spec.package_name, module=module)
    metadata = root / '.minecraft_ai'
    metadata.mkdir(exist_ok=True)
    (metadata / 'complete-proposal.json').write_text(json.dumps(complete.to_dict(), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return (root, complete, module)

def test_generator_emits_only_bounded_async_localhost_boundary(tmp_path: Path) -> None:
    base = _base_proposal()
    root = tmp_path / 'project'
    FabricProjectGenerator().generate(base.spec, root)
    module = _module()
    receipt = generate_local_ai_sidecar(project_root=root, mod_id=base.spec.mod_id, package_name=base.spec.package_name, module=module)
    source_path = root / local_ai_sidecar_source_path(base.spec.package_name, module.module_id)
    manifest_path = root / local_ai_sidecar_manifest_path(module.module_id)
    source = source_path.read_text(encoding='utf-8')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert receipt['status'] == 'GENERATED'
    assert 'http://127.0.0.1:18765/v1/mmm/infer' in source
    assert 'sendAsync(' in source
    assert 'HttpClient.Redirect.NEVER' in source
    assert 'implements HttpResponse.BodySubscriber<byte[]>' in source
    assert 'subscription.cancel()' in source
    assert 'length > limit - buffer.size()' in source
    assert 'InputStream' not in source
    assert 'Semaphore(3)' in source
    assert '.join(' not in source
    assert 'net.minecraft' not in source
    assert 'must-never-enter-the-jar' not in source
    assert 'System.getProperty("mmm.sidecar.token")' in source
    assert 'System.getenv("MMM_SIDECAR_TOKEN")' in source
    assert manifest['source']['validation'] == 'exact_reconstruction_required'
    assert manifest['network']['redirects'] == 'disabled'
    assert manifest['authority']['minecraft_world_mutation'] == 'none'
    assert manifest['secrets']['embedded'] is False

def test_validator_allows_only_exact_approved_source_and_manifest(tmp_path: Path) -> None:
    root, complete, module = _generate_approved_project(tmp_path / 'project')
    spec = complete.base_proposal.spec
    validator = ProjectValidator()
    assert validator.validate(root, spec).passed
    source_path = root / local_ai_sidecar_source_path(spec.package_name, module.module_id)
    source_path.write_text(source_path.read_text(encoding='utf-8') + '// unreviewed edit\n', encoding='utf-8')
    edited = validator.validate(root, spec)
    edited_codes = {finding.code for finding in edited.findings}
    assert 'FORBIDDEN_JAVA_API' in edited_codes
    assert 'LOCAL_AI_SIDECAR_SOURCE_MISMATCH' in edited_codes

def test_manifest_tamper_revokes_network_exception(tmp_path: Path) -> None:
    root, complete, module = _generate_approved_project(tmp_path / 'project')
    spec = complete.base_proposal.spec
    manifest_path = root / local_ai_sidecar_manifest_path(module.module_id)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['network']['endpoint'] = 'http://example.invalid/v1/mmm/infer'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    report = ProjectValidator().validate(root, spec)
    codes = {finding.code for finding in report.findings}
    assert 'FORBIDDEN_JAVA_API' in codes
    assert 'LOCAL_AI_SIDECAR_MANIFEST_MISMATCH' in codes

def test_extra_java_network_code_remains_forbidden(tmp_path: Path) -> None:
    root, complete, _module_value = _generate_approved_project(tmp_path / 'project')
    spec = complete.base_proposal.spec
    extra = root / 'src/main/java' / Path(*spec.package_name.split('.')) / 'integration/UnapprovedNetwork.java'
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(f'package {spec.package_name}.integration;\nimport java.net.Socket;\npublic final class UnapprovedNetwork {{}}\n', encoding='utf-8')
    report = ProjectValidator().validate(root, spec)
    assert any((finding.code == 'FORBIDDEN_JAVA_API' and finding.path.endswith('UnapprovedNetwork.java') for finding in report.findings))

def test_complete_orchestrator_dispatches_reviewed_sidecar_without_model_router(tmp_path: Path) -> None:
    module = _module()
    proposal = _approved_complete(module)
    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / 'out', router_factory=lambda: (_ for _ in ()).throw(AssertionError('reviewed sidecar must not use the custom-code model router')))
    result = orchestrator.execute(proposal, approval_hash=proposal.calculate_hash(), run_name='sidecar_dispatch', options=CompleteExecutionOptions(source_only=True, run_jdt=False, run_blockbench=False, run_runtime=False, run_client=False, run_mineflayer=False, run_visual_review=False))
    assert result.status == 'SOURCE_READY'
    receipts = [receipt for receipt in result.module_receipts if receipt.get('schema_version') == 'mmm/local-ai-sidecar-generation-v1']
    assert len(receipts) == 1
    assert receipts[0]['policy_enforcement'] == 'exact_source_and_manifest_reconstruction'
    assert result.source_validation['status'] == 'PASS'
