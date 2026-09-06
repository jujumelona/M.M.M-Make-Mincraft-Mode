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
