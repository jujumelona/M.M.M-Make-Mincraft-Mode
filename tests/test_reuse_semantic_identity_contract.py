"""Regression for safe official-source routing and semantic cache identity."""
from minecraft_mod_ai.curseforge_semantic_source_review import (
    direct_source_identity, _semantic_cache_key,
)
from minecraft_mod_ai.research_reuse_candidates import (
    _source_link, merge_repository_candidates,
)


def test_nested_gitlab_repo_is_not_aliased_to_subgroup():
    source = "https://gitlab.com/team/platform/teleport-mod/-/tree/main"
    assert direct_source_identity(source) == (
        "gitlab.com", "team/platform", "teleport-mod"
    )
    card = _source_link({
        "source_id": "curseforge:81",
        "provider": "curseforge",
        "source_url": source,
    })
    assert card["repository"] == "team/platform/teleport-mod"
    assert merge_repository_candidates((), (card,)) == [card]


def test_semantic_review_cache_preserves_agent_patient_order():
    params = ("a" * 40, "https://github.com/team/project", ("chase",))
    a = _semantic_cache_key("cats chase dogs", params[0], params[1], params[2])
    b = _semantic_cache_key("dogs chase cats", params[0], params[1], params[2])
    assert a != b
    assert a == _semantic_cache_key(" cats   chase   dogs ", *params)


def test_nested_gitlab_identity_survives_source_manifest_verification():
    import hashlib
    from minecraft_mod_ai.source_transplant import (
        DonorFile, DonorSlice, validate_donor_slice_manifest,
    )
    payload = b"package donor; public class Teleport {}"
    file = DonorFile(
        path="src/main/java/donor/Teleport.java",
        blob_sha="a" * 40,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload), symbols=("Teleport",),
    )
    donor = DonorSlice(
        capability="teleport", repository="team/platform/teleport-mod",
        commit_sha="b" * 40, license_id="MIT",
        source_url="https://gitlab.com/team/platform/teleport-mod/-/tree/main",
        target_compatibility="adapt", files=(file,),
        seed_files=(file.path,), source_symbols=("Teleport",),
        required_dependencies=(), donor_tests=(), confidence=0.9,
    )
    validate_donor_slice_manifest(donor)
    assert donor.repository == "team/platform/teleport-mod"
