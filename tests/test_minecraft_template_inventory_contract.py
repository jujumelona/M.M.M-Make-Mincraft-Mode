from minecraft_mod_ai.minecraft_template_steps import responsibility_ids_for_artifact, steps_for_artifact
from minecraft_mod_ai.structural_routing_contract import CANONICAL_ARTIFACT_KINDS
from minecraft_mod_ai.task_template_catalog import load_template


def test_every_canonical_minecraft_artifact_has_one_sequence_manifest_and_executable_leaves():
    for kind in CANONICAL_ARTIFACT_KINDS:
        manifest_id = f"minecraft/{kind}"
        manifest = load_template(manifest_id)
        assert manifest["id"] == manifest_id
        assert manifest["execution"] == "sequence"

        leaf_ids = responsibility_ids_for_artifact(kind)
        assert leaf_ids
        assert tuple(manifest["steps"]) == leaf_ids

        compiled = steps_for_artifact(kind)
        assert len(compiled) == len(leaf_ids)
        assert tuple(step.template_id for step in compiled) == leaf_ids
        for step in compiled:
            assert step.consumes
            assert step.provides
            assert step.anchor_kinds


def test_generation_contract_is_not_an_artifact_manifest():
    contract = load_template("minecraft/generation_contract")
    assert contract["execution"] == "contract"
    assert "steps" not in contract
    assert "generation_contract" not in CANONICAL_ARTIFACT_KINDS
