"""The Java dependency index follows the generator's optional publication contract."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.artifact_graph_executor import (
    ArtifactGraphError, _declared_java_source_ports, execute_artifact_graph,
)
from minecraft_mod_ai.artifact_job import ArtifactJob
from minecraft_mod_ai.implementation_identity import ExecutorType


def _callback(produces=()):
    return ArtifactJob(
        job_id="feature.callback", template_id="", owner_module="feature",
        executor_type=ExecutorType.PYTHON_GENERATOR,
        canonical_leaf="minecraft/event/callback", produces=produces,
        deterministic_inputs={"_canonical_inputs": {"event_callback_input": {
            "language": "java", "target_path": "src/main/java/example/Feature.java",
        }}},
    )


def test_unpublished_java_job_has_no_dependency_ports():
    assert _declared_java_source_ports([_callback()]) == {}


def test_published_java_port_uses_manifest_output_type_and_order():
    assert _declared_java_source_ports([
        _callback(("feature.receipt", "feature.source")),
    ]) == {"feature.source": "src/main/java/example/Feature.java"}


@pytest.mark.parametrize("ports", [("source",), ("receipt", "source", "extra")])
def test_partial_output_publication_is_rejected_before_generation(ports):
    with pytest.raises(ArtifactGraphError, match="ARTIFACT_GENERATOR_OUTPUT_PORT_ARITY.*feature.callback"):
        _declared_java_source_ports([_callback(ports)])


def test_graph_runs_unpublished_java_job_instead_of_failing_in_dependency_index(monkeypatch):
    monkeypatch.setattr(
        "minecraft_mod_ai.resolved_version_context.execution_context",
        lambda _context, _job: None,
    )
    executed = []

    def generate(job, **kwargs):
        executed.append(job.job_id)
        assert kwargs["context"]["_artifact_java_source_ports"] == {}
        return {"status": "PASS"}

    monkeypatch.setattr("minecraft_mod_ai.artifact_graph_executor._execute_one", generate)
    result = execute_artifact_graph([_callback()])
    assert result["status"] == "PASS"
    assert result["completed_jobs"] == executed == ["feature.callback"]
