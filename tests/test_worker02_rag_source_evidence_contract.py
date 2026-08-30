from __future__ import annotations

import pytest

from minecraft_mod_ai import pre_design_rag_quality_contract as quality


QUERY = "minecraft colony persistent state source"
DOMAIN = {
    "domain_id": "request",
    "requirements": ["Build a persistent colony system."],
    "queries": [QUERY],
}
OBLIGATIONS = (
    {
        "requirement_id": "req-colony",
        "queries": [QUERY],
    },
)


def _record(
    *,
    content: str = "colony persistent state storage implementation",
    source_type: str = "github_source_code",
    source_id: str = "github:example/repo:path/File.java",
    **extra,
):
    record = {
        "source_id": source_id,
        "source_type": source_type,
        "content": content,
    }
    record.update(extra)
    return record


def _grounded(records, **row_extra):
    row = {
        "query": QUERY,
        "evidence_records": list(records),
        "github_provider_status": "available",
    }
    row.update(row_extra)
    return {"queries": [row]}


@pytest.fixture(autouse=True)
def _fixed_obligations(monkeypatch):
    monkeypatch.setattr(
        quality,
        "_approved_requirement_query_obligations",
        lambda domain: OBLIGATIONS,
    )


def _assert_insufficient(grounded):
    with pytest.raises(
        ValueError,
        match="pre-design evidence is insufficient for approved requirements",
    ):
        quality._requirement_evidence_sufficiency(DOMAIN, grounded)


def test_zero_byte_body_is_not_evidence():
    _assert_insufficient(_grounded([_record(content="   ")]))


def test_metadata_or_snippet_only_is_not_evidence():
    record = _record(content="")
    record.pop("content")
    record["snippet"] = "colony persistent state implementation"
    record["metadata_only"] = True
    _assert_insufficient(_grounded([record]))


def test_modrinth_discovery_description_is_not_source_evidence():
    _assert_insufficient(
        _grounded(
            [
                _record(
                    source_type="modrinth_project",
                    source_id="modrinth:example",
                    content="colony persistent state source implementation",
                )
            ]
        )
    )


@pytest.mark.parametrize(
    "flag",
    [
        "content_omitted",
        "source_content_omitted",
        "content_truncated",
        "truncated",
        "text_truncated",
        "payload_truncated",
        "queue_truncated",
    ],
)
def test_omitted_or_truncated_body_fails_closed(flag):
    _assert_insufficient(_grounded([_record(**{flag: True})]))


@pytest.mark.parametrize(
    "flag",
    ["pagination_incomplete", "tree_truncated", "request_budget_exhausted"],
)
def test_incomplete_acquisition_fails_closed(flag):
    _assert_insufficient(_grounded([_record()], **{flag: True}))


@pytest.mark.parametrize("flag", ["body_retrieved", "raw_retrieved", "blob_retrieved"])
def test_github_body_or_raw_not_retrieved_is_not_evidence(flag):
    _assert_insufficient(_grounded([_record(**{flag: False})]))


@pytest.mark.parametrize(
    "error",
    [
        "HTTP 403 Forbidden",
        "HTTP 429 Too Many Requests",
        "TimeoutError: source request timed out",
    ],
)
def test_transport_failure_does_not_pass_with_cached_looking_content(error):
    _assert_insufficient(_grounded([_record()], retrieval_errors=[error]))


def test_zero_results_is_insufficient():
    _assert_insufficient(_grounded([]))


def test_irrelevant_body_is_insufficient():
    _assert_insufficient(
        _grounded([_record(content="weather particles rendering color palette")])
    )


def test_complete_relevant_source_body_passes():
    receipt = quality._requirement_evidence_sufficiency(
        DOMAIN,
        _grounded([_record()]),
    )

    assert receipt is not None
    assert receipt["sufficient"] is True
    assert receipt["validation_version"] == 2
    assert receipt["requirements"][0]["queries_with_content"] == [QUERY]
    assert receipt["query_evidence_receipts"][0]["usable_source_body_ids"] == [
        "github:example/repo:path/File.java"
    ]


def test_round_limit_cannot_false_pass():
    _assert_insufficient(_grounded([_record()], round_limit_reached=True))


def test_requirement_gate_consumes_actual_body_not_matching_snippet():
    misleading = _record(content="weather particles rendering color palette")
    misleading["snippet"] = "colony persistent state source implementation"
    _assert_insufficient(_grounded([misleading]))

    receipt = quality._requirement_evidence_sufficiency(
        DOMAIN,
        _grounded([_record(content="colony persistent state source implementation")]),
    )
    assert receipt is not None
    assert receipt["sufficient"] is True


def test_missing_provenance_is_not_evidence():
    _assert_insufficient(_grounded([_record(source_id="")]))


def test_nested_retrieval_incompleteness_is_fail_closed():
    _assert_insufficient(
        _grounded(
            [
                _record(
                    external_rag={
                        "status": "available",
                        "github_retrieval": {
                            "saturation_reason": "request_budget_exhausted"
                        },
                    }
                )
            ]
        )
    )


def test_fusion_boundary_blocks_metadata_before_model_consumption():
    record = _record(content="")
    record.pop("content")
    record["snippet"] = "colony persistent state source implementation"
    record["source_type"] = "github_search_result"
    with pytest.raises(
        ValueError,
        match="pre-design evidence is insufficient for approved requirements",
    ):
        quality.fuse_grounded_domain_evidence(DOMAIN, _grounded([record]))


def test_fusion_boundary_preserves_actual_body_for_model_consumption():
    body = "colony persistent state source implementation"
    fused = quality.fuse_grounded_domain_evidence(
        DOMAIN,
        _grounded([_record(content=body)]),
    )

    records = fused["queries"][0]["evidence_records"]
    assert records
    assert records[0]["content"] == body
    assert fused["requirement_sufficiency"]["sufficient"] is True
    assert fused["requirement_sufficiency"]["evidence_validation"] == "verified_source_body"
