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
