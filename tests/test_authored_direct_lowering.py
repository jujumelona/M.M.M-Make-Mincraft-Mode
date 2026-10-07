import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules


def test_untyped_authored_plan_is_rejected_before_production() -> None:
    text = (
        "# behavior_contract\nTrade ore.\n"
        "# state_model\nPlayerCredits, Ship, ShipPart, persistence.\n"
        "# verification\nTest transactions."
    )
    plan = AuthoredPlan("space economy", text)

    with pytest.raises(ValueError, match="TYPED_PLAN_REQUIRED"):
        _compile_new_authored_modules(
            plan,
            mod_id="authored_test",
            package_name="example",
            target={
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "mappings": "none",
            },
        )


def test_not_reviewed_python_generator_is_candidate_content_route() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_production import _fact_artifact_route_ready
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.prompt_fact_types import FactType

    leaf = "minecraft/screen/registration"
    facts = build_version_facts("26.2", base_facts={})
    binding = facts["leaf_bindings"][leaf]
    assert binding["state"] == "not_reviewed"
    assert binding["implementation"]["implementation_id"].startswith("python_generator:")

    resolved = SimpleNamespace(
        leaf_bindings={leaf: binding},
        context_id="candidate-26.2",
    )
    fact = ImplementationFact(
        fact_id="market_ui.exists",
        fact_type=FactType.GUI_EXISTS,
        subject="market_ui",
        source_clause="A server-authoritative market screen is required.",
    )

    assert _fact_artifact_route_ready(fact, resolved)

def test_gui_content_relation_does_not_create_item_initializer_dependency() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
    from minecraft_mod_ai.artifact_job import validate_artifact_job_graph
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.prompt_fact_types import FactType

    host_facts = build_version_facts("26.2", base_facts={})
    resolved = SimpleNamespace(
        minecraft="26.2",
        context_id="candidate-26.2",
        leaf_bindings=host_facts["leaf_bindings"],
    )
    gui = ImplementationFact(
        fact_id="gui_market.exists",
        fact_type=FactType.GUI_EXISTS,
        subject="gui_market",
        source_clause="Market screen exists.",
    )
    relation = ImplementationFact(
        fact_id="gui_market.requires.gui_ship",
        fact_type=FactType.CONTENT_RELATION,
        subject="gui_market",
        object="gui_ship",
        value={"relation": "requires"},
        source_clause="Market screen requires ship screen.",
    )

    jobs = expand_facts_to_jobs(
        (gui, relation),
        mod_id="space_mod",
        package_name="example.space",
        minecraft_version="26.2",
        version_context=resolved,
    )

    assert len(jobs) == 1
    assert jobs[0].canonical_leaf == "minecraft/screen/registration"
    assert jobs[0].template_id == ""
    assert all(job.template_id != "fabric/item/initializer" for job in jobs)
    validate_artifact_job_graph(jobs, module_ids={"gui_market"})

