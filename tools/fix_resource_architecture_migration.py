from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _patch_proposal_deserialization_installer() -> None:
    path = ROOT / "minecraft_mod_ai/proposal_deserialization_contract.py"
    text = path.read_text(encoding="utf-8")
    if (
        "def install(spec_module:" in text
        and "'install_proposal_deserialization_contracts'" in text
        and "'install'" in text.split("__all__ =", 1)[-1]
    ):
        return
    old = "__all__ = ['install_proposal_deserialization_contracts']\n"
    new = '''def install(spec_module: object, complete_spec_module: object) -> None:\n    """Bind the centralized strict deserializer to the public spec modules."""\n    from .capabilities import capability_manifest_hash\n    from .knowledge import evidence_snapshot_hash\n\n    install_proposal_deserialization_contracts(\n        proposal_cls=spec_module.Proposal,\n        proposal_status_cls=spec_module.ProposalStatus,\n        spec_validation_error=spec_module.SpecValidationError,\n        content_spec_cls=spec_module.ContentSpec,\n        content_kind_cls=spec_module.ContentKind,\n        boss_spec_cls=spec_module.BossSpec,\n        mod_spec_cls=spec_module.ModSpec,\n        deferred_request_cls=spec_module.DeferredRequest,\n        evidence_source_cls=spec_module.EvidenceSource,\n        capability_manifest_hash=capability_manifest_hash,\n        evidence_snapshot_hash=evidence_snapshot_hash,\n        json_bool=spec_module._json_bool,\n        complete_proposal_cls=complete_spec_module.CompleteProposal,\n        complete_proposal_status_cls=complete_spec_module.CompleteProposalStatus,\n        production_module_cls=complete_spec_module.ProductionModule,\n        asset_request_cls=complete_spec_module.AssetRequest,\n    )\n\n\n__all__ = ['install', 'install_proposal_deserialization_contracts']\n'''
    if text.count(old) != 1:
        raise RuntimeError("proposal deserialization installer anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _patch_resource_architecture_test() -> None:
    path = ROOT / "tests/test_resource_architecture.py"
    text = path.read_text(encoding="utf-8")
    if "def test_obsolete_backend_prompt_manifests_are_removed()" in text:
        return
    old = '''def test_asset_manifests_do_not_embed_lora_trigger() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        text = (ROOT / "minecraft_mod_ai/templates" / relative).read_text(encoding="utf-8")\n        assert "PixArFK" not in text\n        assert "Pixel Art" not in text\n'''
    new = '''def test_obsolete_backend_prompt_manifests_are_removed() -> None:\n    for relative in ("asset/block_tile.yaml", "asset/item_sprite.yaml", "asset/entity_texture.yaml", "asset/gui_panel.yaml"):\n        assert not (ROOT / "minecraft_mod_ai/templates" / relative).exists()\n\n    validation = (ROOT / "minecraft_mod_ai/template_contract_validation.py").read_text(encoding="utf-8")\n    for identifier in ("asset/block_tile", "asset/item_sprite", "asset/entity_texture", "asset/gui_panel"):\n        assert identifier not in validation\n'''
    if text.count(old) != 1:
        raise RuntimeError("resource architecture obsolete-manifest test anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _patch_feature_semantic_convergence() -> None:
    path = ROOT / "minecraft_mod_ai/feature_template_pipeline.py"
    text = path.read_text(encoding="utf-8")
    if "max_depth" not in text and "parent_failed_checks" in text:
        return

    old_header = '''    checkpoint=None,\n    depth=0,\n    max_depth=32,\n    ancestry=(),\n    ancestry_signatures=(),\n):\n    """Recursively split until every leaf is atomic or semantic progress stops."""\n    if depth > max_depth:\n        raise TemplateBlocked("FEATURE_DECOMPOSE: maximum decomposition depth exceeded")\n\n'''
    new_header = '''    checkpoint=None,\n    ancestry=(),\n    ancestry_signatures=(),\n    parent_failed_checks=None,\n):\n    """Recursively split while the finite unresolved atomicity set strictly decreases."""\n\n'''
    if text.count(old_header) != 1:
        raise RuntimeError("feature semantic convergence header anchor mismatch")
    text = text.replace(old_header, new_header, 1)

    old_atomicity = '''    atomicity = _run_atomic_checks(\n        router,\n        completed,\n        allowed_refs=allowed_refs,\n        progress=progress,\n        checkpoint=checkpoint,\n    )\n    node = {\n'''
    new_atomicity = '''    atomicity = _run_atomic_checks(\n        router,\n        completed,\n        allowed_refs=allowed_refs,\n        progress=progress,\n        checkpoint=checkpoint,\n    )\n    current_failed = frozenset(atomicity["failed_checks"])\n    if parent_failed_checks is not None:\n        previous_failed = frozenset(parent_failed_checks)\n        if current_failed and not current_failed < previous_failed:\n            raise TemplateBlocked(\n                "FEATURE_DECOMPOSE: unresolved atomic checks did not strictly decrease"\n            )\n    node = {\n'''
    if text.count(old_atomicity) != 1:
        raise RuntimeError("feature semantic convergence atomicity anchor mismatch")
    text = text.replace(old_atomicity, new_atomicity, 1)

    old_recursive = '''                checkpoint=checkpoint,\n                depth=depth + 1,\n                max_depth=max_depth,\n                ancestry=next_ancestry,\n                ancestry_signatures=next_signatures,\n'''
    new_recursive = '''                checkpoint=checkpoint,\n                ancestry=next_ancestry,\n                ancestry_signatures=next_signatures,\n                parent_failed_checks=atomicity["failed_checks"],\n'''
    if text.count(old_recursive) != 1:
        raise RuntimeError("feature semantic convergence recursion anchor mismatch")
    path.write_text(text.replace(old_recursive, new_recursive, 1), encoding="utf-8")


def _patch_template_runtime_contract_tests() -> None:
    path = ROOT / "tests/test_template_runtime_ssot.py"
    text = path.read_text(encoding="utf-8")

    old_record = '''def test_record_execution_has_no_model_owned_status_loop() -> None:\n    bounded = (PKG / "bounded_record_template.py").read_text(encoding="utf-8")\n    batch = (PKG / "task_template_batch_runner.py").read_text(encoding="utf-8")\n    assert "record/done" in bounded\n    assert "applicability" in bounded\n    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(bounded)))\n    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(batch)))\n    assert '\"status\"' not in batch\n'''
    new_record = '''def test_record_execution_has_no_model_owned_status_loop() -> None:\n    bounded = (PKG / "bounded_record_template.py").read_text(encoding="utf-8")\n    batch = (PKG / "task_template_batch_runner.py").read_text(encoding="utf-8")\n    assert "record_cardinality_response_schema" in bounded\n    assert '\"count\"' in bounded\n    assert "record/done" not in bounded\n    assert "applicability" in bounded\n    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(bounded)))\n    assert not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(batch)))\n    assert '\"status\"' not in batch\n'''
    if old_record in text:
        text = text.replace(old_record, new_record, 1)
    elif new_record not in text:
        raise RuntimeError("record cardinality runtime test anchor mismatch")

    old_feature = '''def test_feature_convergence_is_semantic_not_depth_limited() -> None:\n    text = (PKG / "feature_template_pipeline.py").read_text(encoding="utf-8")\n    assert "max_depth" not in text\n    assert "semantic ancestry cycle/no-progress" in text\n    assert "semantically duplicate children" in text\n'''
    new_feature = '''def test_feature_convergence_is_semantic_not_depth_limited() -> None:\n    text = (PKG / "feature_template_pipeline.py").read_text(encoding="utf-8")\n    assert "max_depth" not in text\n    assert "semantic ancestry cycle/no-progress" in text\n    assert "semantically duplicate children" in text\n    assert "unresolved atomic checks did not strictly decrease" in text\n'''
    if old_feature in text:
        text = text.replace(old_feature, new_feature, 1)
    elif new_feature not in text:
        raise RuntimeError("feature convergence runtime test anchor mismatch")

    old_response = '''def test_response_contracts_are_not_python_hardcoded() -> None:\n    text = (PKG / "model_response_templates.py").read_text(encoding="utf-8")\n    assert "_TEMPLATES" not in text\n    assert "response/contracts.json" in text\n    assert (TEMPLATES / "response" / "contracts.json").is_file()\n'''
    new_response = '''def test_response_contracts_are_not_python_hardcoded() -> None:\n    text = (PKG / "model_response_templates.py").read_text(encoding="utf-8")\n    assert "_TEMPLATES" not in text\n    assert 'RUNTIME_TEMPLATE_ROOT / "response" / "contracts.json"' in text\n    assert (TEMPLATES / "response" / "contracts.json").is_file()\n'''
    if old_response in text:
        text = text.replace(old_response, new_response, 1)
    elif new_response not in text:
        raise RuntimeError("response contract runtime test anchor mismatch")

    path.write_text(text, encoding="utf-8")


def _patch_recursive_template_package_data() -> None:
    path = ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    if all(pattern in text for pattern in (
        '"templates/**/*.yaml"',
        '"templates/**/*.json"',
        '"templates/**/*.java.fmt"',
    )):
        return
    old = '''  "templates/*.yaml",\n  "templates/*/*.yaml",\n  "templates/*/*/*.yaml",\n  "templates/*/*/*/*.yaml",\n  "templates/*.json",\n  "templates/*/*.json",\n  "templates/*/*/*.json",\n  "templates/*/*/*/*.json",\n  "templates/*.java.fmt",\n  "templates/*/*.java.fmt",\n  "templates/*/*/*.java.fmt",\n  "templates/*/*/*/*.java.fmt",\n'''
    new = '''  "templates/**/*.yaml",\n  "templates/**/*.json",\n  "templates/**/*.java.fmt",\n'''
    if text.count(old) != 1:
        raise RuntimeError("recursive package-data anchor mismatch")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    _patch_proposal_deserialization_installer()
    _patch_resource_architecture_test()
    _patch_feature_semantic_convergence()
    _patch_template_runtime_contract_tests()
    _patch_recursive_template_package_data()


if __name__ == "__main__":
    main()
