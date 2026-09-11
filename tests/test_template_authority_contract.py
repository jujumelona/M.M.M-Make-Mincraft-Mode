import pytest

from minecraft_mod_ai import minecraft_template_steps
from minecraft_mod_ai import task_template_catalog


def test_runtime_template_root_is_package_owned():
    expected = task_template_catalog.Path(task_template_catalog.__file__).with_name("templates").resolve()
    assert task_template_catalog.RUNTIME_TEMPLATE_ROOT == expected
    assert task_template_catalog.ROOT == expected


@pytest.mark.parametrize(
    "identifier",
    [
        "../templates/prompt/capture",
        "/minecraft/block",
        "minecraft\\block",
        "minecraft/block.yaml",
        " minecraft/block",
        "minecraft/block ",
        "minecraft//block",
    ],
)
def test_template_identifier_rejects_noncanonical_paths(identifier):
    with pytest.raises(ValueError, match="TEMPLATE_PATH"):
        task_template_catalog.load_template(identifier)


def test_template_identifier_requires_string():
    with pytest.raises(ValueError, match="identifier must be a string"):
        task_template_catalog.load_template(None)


def test_missing_runtime_template_has_explicit_error():
    with pytest.raises(ValueError, match="TEMPLATE_MISSING"):
        task_template_catalog.load_template("minecraft/definitely_missing_contract")


def test_manifest_and_leaf_have_distinct_authority_roles():
    manifest = task_template_catalog.load_template("minecraft/block")
    leaf = task_template_catalog.load_template("minecraft/block/registry")

    assert manifest["id"] == "minecraft/block"
    assert manifest["execution"] == "sequence"
    assert "minecraft/block/registry" in manifest["steps"]
    assert leaf["id"] == "minecraft/block/registry"
    assert "execution" not in leaf
    assert leaf["provides"] == ["registry"]


def test_minecraft_step_preserves_leaf_consumes_and_provides():
    steps = minecraft_template_steps.steps_for_artifact("block")
    registry = next(step for step in steps if step.template_id == "minecraft/block/registry")

    assert "artifact_requirement" in registry.consumes
    assert "target_cell" in registry.consumes
    assert "minecraft/block/registry:complete" == registry.provides[0]
    assert "registry" in registry.provides
    assert registry.anchor_kinds == ("registry_id", "symbol", "test")


def test_minecraft_sequence_keeps_explicit_completion_chain():
    steps = minecraft_template_steps.steps_for_artifact("block")
    assert len(steps) > 1
    for previous, current in zip(steps, steps[1:]):
        assert previous.provides[0] in current.consumes
