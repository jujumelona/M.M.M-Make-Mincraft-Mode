"""Published candidate bindings must match the executors shipped beside them."""
from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.host_version_catalog import load_host_catalog
from minecraft_mod_ai.integrity_dispatcher import verify_job_binding


def test_published_screen_candidates_enter_runtime_with_current_identity():
    _, bundles = load_host_catalog()
    for bundle in bundles:
        leaf = "minecraft/screen/registration"
        binding = bundle.leaf_bindings[leaf]
        assert binding["state"] == "not_reviewed"
        job = ArtifactJob(
            "published-screen", "", "market_ui", canonical_leaf=leaf,
            implementation_id=binding["implementation"]["implementation_id"],
        )
        verify_job_binding(job, bundle, {})


def test_all_published_leaf_owned_hashes_match_current_authority():
    from minecraft_mod_ai.integrity_bootstrap import get_integrity_authority

    authority = get_integrity_authority()
    authority.verify_live()
    _, bundles = load_host_catalog()
    for bundle in bundles:
        for leaf, binding in bundle.leaf_bindings.items():
            if binding["state"] == "unsupported":
                continue
            impl = binding["implementation"]
            actual = authority.implementations.get_implementation(impl["implementation_id"])
            validator = authority.implementations.get_validator(impl["validator_profile"])
            expected = {
                "implementation_sha256": actual.content_sha256,
                "validator_sha256": validator.source_hash,
                "input_schema_sha256": authority.types.get_schema_hash(leaf + ":input"),
                "output_schema_sha256": authority.types.get_schema_hash(leaf + ":output"),
            }
            for field, digest in expected.items():
                assert impl[field] == digest, (bundle.minecraft, leaf, field)
