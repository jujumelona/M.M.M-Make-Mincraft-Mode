from __future__ import annotations


def test_season_repository_candidate_is_not_false_negative():
    from minecraft_mod_ai import pre_design_external_source_contract as external

    query = "minecraft seasonal crop planting mod"
    repository = {
        "full_name": "lucaargolo/fabric-seasons",
        "description": "A Fabric mod that adds seasons to Minecraft",
        "topics": ["minecraft", "fabric", "mod", "seasons"],
    }
    assert external._repository_candidate_relevant(query, repository)
    assert external._body_relevant(
        query,
        "Fabric Seasons is a mod for Minecraft that adds seasons and seasonal world behavior.",
    )


def test_empty_evidence_projection_never_calls_small_model():
    from minecraft_mod_ai import small_model_predesign_research as research

    class NeverRouter:
        def generate_text(self, *args, **kwargs):
            raise AssertionError("model must not be called for an empty evidence projection")

    class ProjectRag:
        @staticmethod
        def _read_evidence_pages(document):
            raise AssertionError("empty evidence pages must not be read")

        @staticmethod
        def _prompt_document_receipt(document):
            return {"page_count": int(document.get("page_count") or 0)}

    document = {
        "domain_id": "request",
        "model_unit_count": 0,
        "page_count": 1,
        "document_sha256": "sha256:empty",
    }
    result = research.research_document_domain(
        object(),
        ProjectRag(),
        NeverRouter(),
        prompt="계절 작물과 요리",
        domain={"domain_id": "request", "objective": "seasonal crops"},
        document=document,
        trace_metadata=None,
    )
    assert result["claims"] == []
    assert result["research_evidence_status"] == "no_relevant_external_evidence"
    assert "no_claim_bearing_source_bodies" in result["page_local_diagnostics"]
