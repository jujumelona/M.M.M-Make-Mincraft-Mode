from __future__ import annotations


def test_removed_github_donor_research_cannot_reenter_predesign() -> None:
    from pathlib import Path

    from minecraft_mod_ai.pre_design_research_pipeline import _pre_design_brief

    root = Path(__file__).resolve().parents[1] / "minecraft_mod_ai"
    assert not (root / "pre_design_external_source_contract.py").exists()
    brief = _pre_design_brief("minecraft seasonal crop planting")
    providers = set(brief["domains"][0]["providers"])
    assert {"official_docs", "project_rag"} <= providers
    assert not providers.intersection({"github", "curseforge", "modrinth"})
    assert [domain["domain_id"] for domain in brief["domains"]] == ["request"]


def test_corrective_queries_are_host_owned_and_do_not_call_model():
    from minecraft_mod_ai import pre_design_rag_corrective as corrective

    class NeverModel:
        def __getattr__(self, name):
            raise AssertionError(f"model/project helper called: {name}")

    seen = {"minecraft alien mob entity mod"}
    result = corrective._generate_gap_queries(
        NeverModel(),
        NeverModel(),
        NeverModel(),
        domain={
            "domain_id": "request",
            "queries": [
                "minecraft alien mob entity mod",
                "minecraft colony settlement building mod",
            ],
        },
        gaps=["colonization evidence missing"],
        prior_queries=["minecraft alien mob entity mod"],
        seen=seen,
        raw_prompt="우주 식민지화 모드",
        progress_label="test",
    )
    assert result == ["minecraft colony settlement building mod"]
