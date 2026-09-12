import pytest

from minecraft_mod_ai import minecraft_template_steps
from minecraft_mod_ai import task_template_catalog
from minecraft_mod_ai.host_version_catalog import host_target, load_host_catalog
from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.product_support_matrix import SUPPORTED_MINECRAFT_VERSIONS
from minecraft_mod_ai.version_template_context import resolved_template_facts


_REQUIRED_TARGET_FIELDS = (
    "minecraft_version",
    "loader",
    "java_version",
    "fabric_loader",
    "fabric_api",
    "fabric_loom",
    "gradle",
    "gradle_sha256",
    "data_pack_version",
    "resource_pack_version",
    "resource_pack_format",
    "release_metadata_url",
)
_HOST_FACT_KEYS = (
    "host_revision",
    "capabilities",
    "api_symbols",
    "schemas",
    "artifact_rules",
    "dependency_coordinates",
    "repositories",
    "replacements",
    "leaf_bindings",
)
_IMAGE_PROVIDER_IDENTITY_FIELDS = (
    "model_id",
    "lora_model_id",
    "lora_weight_name",
    "lora_adapter_name",
    "lora_trigger",
)


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


def test_every_supported_version_is_complete_host_authority_before_template_projection():
    _, bundles = load_host_catalog()
    published = {bundle.minecraft: bundle for bundle in bundles}
    missing = sorted(set(SUPPORTED_MINECRAFT_VERSIONS) - set(published))
    assert not missing, f"product-supported versions missing HOST bundles: {missing}"

    for version in SUPPORTED_MINECRAFT_VERSIONS:
        target = host_target(version)
        target.validate()
        resolved = target.version_context
        assert resolved.context_id == published[version].context_id

        public_target = target.public_dict()
        projected = resolved_template_facts(resolved)
        snapshot = resolved.to_dict()

        for field in _REQUIRED_TARGET_FIELDS:
            value = public_target[field]
            assert value is not None, f"{version}: missing HOST target field {field}"
            if isinstance(value, str):
                assert value.strip(), f"{version}: blank HOST target field {field}"
                assert value.strip().casefold() not in {"auto", "latest", "unresolved", "*"}, (
                    f"{version}: unresolved HOST target field {field}={value!r}"
                )
            assert projected[field] == value, f"{version}: template projection changed HOST field {field}"

        naming = public_target["naming_regime"]
        assert projected["naming_regime"] == naming["kind"]
        assert projected["mappings_applicable"] is naming["mappings_applicable"]
        assert projected["pack_versions"] == public_target["pack_versions"]
        if naming["mappings_applicable"]:
            assert projected["mappings"] == public_target["mappings"]
        else:
            assert "mappings" not in projected

        for key in _HOST_FACT_KEYS:
            assert key in snapshot["host_facts"], f"{version}: missing HOST fact domain {key}"
            assert projected[key] == snapshot["host_facts"][key], (
                f"{version}: template projection changed HOST fact domain {key}"
            )


def test_task_templates_do_not_own_image_provider_identity():
    public_registry = ModelRegistry().to_public_dict()
    provider_literals = set()
    for profile in public_registry["profiles"].values():
        image_role = profile["roles"].get("image_generator", {})
        for field in _IMAGE_PROVIDER_IDENTITY_FIELDS:
            value = image_role.get(field)
            if isinstance(value, str) and value.strip():
                provider_literals.add(value.strip())

    leaks = {}
    for path in task_template_catalog.RUNTIME_TEMPLATE_ROOT.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        found = sorted(literal for literal in provider_literals if literal in text)
        if found:
            leaks[str(path.relative_to(task_template_catalog.RUNTIME_TEMPLATE_ROOT))] = found

    assert not leaks, f"task templates must not own image provider identity: {leaks}"
