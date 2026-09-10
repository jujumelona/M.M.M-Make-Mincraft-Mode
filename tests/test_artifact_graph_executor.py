from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_graph_executor import (
    ArtifactGraphError,
    execute_artifact_graph,
)
from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact


def test_expanded_item_jobs_execute_through_scoped_port_graph():
    jobs = expand_facts_to_jobs(
        [
            PromptFact(
                fact_id="fact_001",
                fact_type=FactType.ITEM_EXISTS,
                subject="raw_lunite",
            ),
            PromptFact(
                fact_id="fact_002",
                fact_type=FactType.ITEM_STACK_LIMIT,
                subject="raw_lunite",
                value=16,
            ),
        ],
        mod_id="space",
        package_name="com.foo.space",
    )
    result = execute_artifact_graph(jobs)
    assert result["status"] == "PASS"
    assert set(result["completed_jobs"]) == {job.job_id for job in jobs}
    assert "raw_lunite.registry_id" in result["ports"]
    assert "raw_lunite.java_symbol" in result["ports"]
    assert "raw_lunite.model_ref" in result["ports"]
    assert "raw_lunite.translation_key" in result["ports"]


def test_graph_rejects_missing_producer_before_running_any_job():
    job = ArtifactJob(
        job_id="raw_lunite.model_basic",
        template_id="fabric/item/model_basic",
        owner_module="raw_lunite",
        requires=("raw_lunite.registry_id",),
        produces=("raw_lunite.model_ref",),
        deterministic_inputs={"mod_id": "space", "registry_path": "raw_lunite"},
    )
    with pytest.raises(ArtifactGraphError, match="ARTIFACT_GRAPH_MISSING_PRODUCER"):
        execute_artifact_graph([job])


def test_graph_rejects_duplicate_producers():
    common = {
        "template_id": "fabric/item/model_basic",
        "owner_module": "raw_lunite",
        "produces": ("raw_lunite.model_ref",),
        "deterministic_inputs": {
            "mod_id": "space",
            "registry_path": "raw_lunite",
        },
    }
    first = ArtifactJob(job_id="a", **common)
    second = ArtifactJob(job_id="b", **common)
    with pytest.raises(ArtifactGraphError, match="ARTIFACT_DUPLICATE_PRODUCER"):
        execute_artifact_graph([first, second])


def test_shared_java_targets_resume_without_duplicate_insertions(tmp_path):
    from minecraft_mod_ai.artifact_materializer import ensure_artifact_scaffolding
    ensure_artifact_scaffolding(tmp_path,mod_id="demo",package_name="org.demo")
    facts=[PromptFact(fact_id=name,fact_type=FactType.ITEM_EXISTS,subject=name) for name in ("first_item","second_item")]
    def jobs():
        return expand_facts_to_jobs(facts,mod_id="demo",package_name="org.demo")
    first=execute_artifact_graph(jobs(),base_dir=tmp_path)
    registry=tmp_path/"src/main/java/org/demo/registry/ModItems.java"
    before=registry.read_bytes()
    second=execute_artifact_graph(jobs(),base_dir=tmp_path)
    assert registry.read_bytes()==before
    assert all(r.get("reuse")=="VERIFIED_CHECKPOINT" for r in second["receipts"])
    assert first["ports"]==second["ports"]


def test_typed_input_rejects_wrong_registry_kind():
    from minecraft_mod_ai.artifact_ports import PortRegistry
    from minecraft_mod_ai.task_template_runner import execute_artifact_template
    jobs=expand_facts_to_jobs([PromptFact(fact_id="one",fact_type=FactType.ITEM_EXISTS,subject="sample")],mod_id="demo",package_name="org.demo")
    model=next(j for j in jobs if j.template_id=="fabric/item/model_basic")
    registry=PortRegistry()
    registry.register("sample.registry_id","demo:sample",kind="REGISTRY_ID",target_type="Block")
    with pytest.raises(ValueError,match="PORT_TARGET_TYPE_MISMATCH"):
        execute_artifact_template(model,port_registry=registry)
