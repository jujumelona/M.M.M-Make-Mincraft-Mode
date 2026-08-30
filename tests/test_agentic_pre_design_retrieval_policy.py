from __future__ import annotations

import json

from minecraft_mod_ai import pre_design_rag_fusion as fusion
from minecraft_mod_ai import pre_design_rag_support as support


def test_fusion_recomposes_large_retrieval_under_byte_budget(monkeypatch) -> None:
    monkeypatch.setenv("MMM_PREDESIGN_EVIDENCE_BYTE_BUDGET", str(12 * 1024))
    monkeypatch.setenv("MMM_PREDESIGN_EVIDENCE_EXCERPT_CHARS", "1200")
    rows = []
    for query_index in range(6):
        query = f"minecraft mechanic topic{query_index} planning"
        records = []
        for record_index in range(12):
            records.append(
                {
                    "source_id": f"source:{query_index}:{record_index}",
                    "source_type": "modrinth_project",
                    "content_sha256": f"sha256:{query_index}:{record_index}",
                    "content": (
                        f"topic{query_index} relevant planning evidence "
                        + ("x" * 5000)
                    ),
                }
            )
        rows.append(
            {
                "query": query,
                "query_sha256": f"q{query_index}",
                "evidence_records": records,
            }
        )

    result = fusion.fuse_grounded_domain_evidence({}, {"queries": rows})
    meta = result["fusion"]
    records = result["queries"][0]["evidence_records"]

    assert meta["unique_record_count"] == 72
    assert meta["selected_record_count"] < meta["unique_record_count"]
    assert meta["selected_content_bytes"] <= meta["evidence_byte_budget"] + 1200
    covered = {
        query
        for record in records
        for query in record["retrieval_fusion"]["matched_queries"]
    }
    assert covered == {row["query"] for row in rows}
    assert all(len(record["content"]) <= 1202 for record in records)


class _Agentic:
    class SpecValidationError(ValueError):
        pass


class _VariantSupportRag:
    @staticmethod
    def _generate_bounded(
        agentic_module,
        router,
        *,
        messages,
        response_schema,
        parser,
        progress_label,
    ):
        del agentic_module, router, messages, progress_label
        # Matches the alternate shape produced by the real local Qwen log:
        # top-level diagnostics + claims[] + quote alias.
        assert response_schema["additionalProperties"] is True
        return parser(
            json.dumps(
                {
                    "sufficient": True,
                    "gaps": [],
                    "claims": [
                        {
                            "claim_index": 0,
                            "supported": True,
                            "quote": "villager charges market price times margin",
                        },
                        {
                            "claim_index": 1,
                            "supported": False,
                            "quote": "",
                        },
                    ],
                }
            )
        )


def test_claim_support_accepts_qwen_alias_shape_but_host_checks_exact_quote() -> None:
    page = {
        "page_ref": "sha256:trade#page=1/1",
        "content": (
            "Trading configuration says the villager charges market price times margin "
            "when selling an item to the player."
        ),
    }
    accepted, rejected = support._verify_page_claims(
        _Agentic,
        _VariantSupportRag,
        object(),
        domain_id="request",
        page=page,
        claims=[
            "Villager sale prices apply a configurable margin.",
            "The same source implements spacecraft upgrades.",
        ],
        progress_label="test",
    )
    assert [item["claim"] for item in accepted] == [
        "Villager sale prices apply a configurable margin."
    ]
    assert rejected == ["The same source implements spacecraft upgrades."]


class _BadQuoteSupportRag:
    @staticmethod
    def _generate_bounded(
        agentic_module,
        router,
        *,
        messages,
        response_schema,
        parser,
        progress_label,
    ):
        del agentic_module, router, messages, response_schema, progress_label
        return parser(
            json.dumps(
                {
                    "claims": [
                        {
                            "claim_index": 0,
                            "supported": True,
                            "quote": "invented quote not present in evidence",
                        }
                    ]
                }
            )
        )


def test_claim_support_never_trusts_alias_without_exact_host_quote() -> None:
    accepted, rejected = support._verify_page_claims(
        _Agentic,
        _BadQuoteSupportRag,
        object(),
        domain_id="request",
        page={"page_ref": "p", "content": "real evidence only"},
        claims=["A claim"],
        progress_label="test",
    )
    assert accepted == []
    assert rejected == ["A claim"]
