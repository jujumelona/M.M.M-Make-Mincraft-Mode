from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.artifact_expansion import _canonical_candidate_inputs
from minecraft_mod_ai.canonical_generators import generate_canonical_leaf
from minecraft_mod_ai.canonical_java_target_contract import assert_canonical_java_target
from minecraft_mod_ai.canonical_schema_compiler import compile_leaf_schemas
from minecraft_mod_ai.host_block_entity_shell import basic_block_entity_candidate_contract
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.project_edit import ensure_fabric_main_entrypoints


def _inputs(version="26.2"):
    inputs = _canonical_candidate_inputs(
        SimpleNamespace(
            source_clause="Register a server-authoritative space dock block entity.",
            display_name="Space Dock Anchor",
            fact_type=SimpleNamespace(value="BLOCK_ENTITY_EXISTS"),
            parent_requirement="",
        ),
        canonical_leaf="minecraft/block_entity/registry",
        mod_id="ships",
        package_name="org.example.ships",
        package_path="org/example/ships",
        subject="space_dock_anchor",
        minecraft_version=version,
        context_id="host-26.2-context",
    )
    inp, _out = compile_leaf_schemas("minecraft/block_entity/registry")
    Draft202012Validator(inp).validate(inputs)
    return inputs


@pytest.mark.parametrize("version", ["26.1.2", "26.2"])
def test_block_entity_java_api_is_26x_mojang_and_host_owned(version):
    spec = _inputs(version)["block_entity_registry_input"]
    assert spec["slots"] == []
    assert spec["bindings"]["host_block_entity_capability"] == "basic_block_entity_with_owned_block"
    assert spec["bindings"]["host_block_entity_entrypoint"].endswith(
        ".SpaceDockAnchorRegistry"
    )
    source = render_template({"render": spec["render_mold"]}, {})
    assert 'Identifier.fromNamespaceAndPath("ships", "space_dock_anchor")' in source
    assert "net.minecraft.resources.Identifier;" in source
    assert "net.minecraft.util.Identifier" not in source
    assert "FabricBlockEntityTypeBuilder.create(GeneratedBlockEntity::new, BLOCK).build()" in source
    assert "new GeneratedBlock(BlockBehaviour.Properties.of().setId(blockKey))" in source
    assert "class GeneratedBlock extends BaseEntityBlock" in source
    assert "class GeneratedBlockEntity extends BlockEntity" in source
    assert "void onInitialize()" in source
    assert "getRenderShape" not in source  # no fabricated renderer contract
    assert_canonical_java_target(source, spec)


def test_full_canonical_leaf_never_invokes_model():
    leaf = "minecraft/block_entity/registry"
    inputs = _inputs()
    inp, out = compile_leaf_schemas(leaf)

    class Authority:
        def validate_input(self, _, obj):
            Draft202012Validator(inp).validate(obj)

        def validate_output(self, _, obj):
            Draft202012Validator(out).validate(obj)

    result = generate_canonical_leaf(
        inputs, leaf_id=leaf, router=None, authority=SimpleNamespace(types=Authority()),
    )
    assert "Registry.register(" in result["block_entity_registry_artifact"]
    assert result["block_entity_registry_receipt"]["content_sha256"].startswith("sha256:")


def test_fabric_main_entrypoints_are_atomic_idempotent_and_preserve_scaffold(tmp_path):
    root = tmp_path / "mod"
    metadata = root / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({
            "schemaVersion": 1, "id": "ships",
            "entrypoints": {"main": ["org.example.ships.ShipsMod"]},
        }),
        encoding="utf-8",
    )
    info = SimpleNamespace(root=root, fabric_mod_json=metadata)
    name = "org.example.ships.generated.SpaceDockAnchorRegistry"
    ensure_fabric_main_entrypoints(info, entrypoints=(name,))
    after = json.loads(metadata.read_text())
    assert after["entrypoints"]["main"] == [
        "org.example.ships.ShipsMod", name,
    ]
    snapshot = metadata.read_bytes()
    receipt = ensure_fabric_main_entrypoints(info, entrypoints=(name,))
    assert receipt["status"] == "UNCHANGED"
    assert metadata.read_bytes() == snapshot


@pytest.mark.parametrize("subject", ["../escape", "UpperCASE", 'unsafe";', ""])
def test_invalid_block_entity_id_fails_before_model_or_materialization(subject):
    with pytest.raises(ValueError, match="HOST_BLOCK_ENTITY_REGISTRY_IDENTITY_INVALID"):
        basic_block_entity_candidate_contract(
            package_name="org.example.ships.generated",
            class_name="DockRegistry",
            mod_id="ships", subject=subject, minecraft_version="26.2",
        )


def test_unreviewed_api_epoch_fails_closed():
    with pytest.raises(ValueError, match="HOST_BLOCK_ENTITY_UNREVIEWED_API_EPOCH"):
        basic_block_entity_candidate_contract(
            package_name="org.example.ships.generated",
            class_name="DockRegistry",
            mod_id="ships", subject="dock", minecraft_version="26.3",
        )
