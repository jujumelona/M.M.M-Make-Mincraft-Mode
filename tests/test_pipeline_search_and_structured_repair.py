from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import platform_optimizer
from minecraft_mod_ai.llama_structured_decode_policy import _structured_repair_request
from minecraft_mod_ai.model_adapters import GenerationRequest
from minecraft_mod_ai.pipeline_hardening_v2 import _search_variants


def test_generated_task_name_gets_bounded_semantic_search_variant() -> None:
    variants = _search_variants("TaskAlienPlanetInteractionSemanticImplementation")
    assert variants[0] == "TaskAlienPlanetInteractionSemanticImplementation"
    assert len(variants) == 2
    assert "Task" not in variants[1]
    assert "Semantic" not in variants[1]
    assert "Implementation" not in variants[1]
    assert "Alien" in variants[1]
    assert "Planet" in variants[1]


def test_mod_search_is_broad_then_exact_metadata_verified() -> None:
    class Client:
        def __init__(self) -> None:
            self.search_calls = []
            self.inspect_calls = []

        def search(self, source, query, **kwargs):
            self.search_calls.append((source, query, dict(kwargs)))
            return {
                "candidates": [
                    {"candidate_id": "modrinth:compatible"},
                    {"candidate_id": "modrinth:other"},
                ]
            }

        def inspect_modrinth_project(self, project_id, *, minecraft_version, loader):
            self.inspect_calls.append((project_id, minecraft_version, loader))
            eligible = project_id == "compatible" and minecraft_version == "1.21.1"
            return {
                "license_policy": "permissive_candidate:mit",
                "versions": [
                    {
                        "eligible_for_selection": eligible,
                        "files": [{"sha512": "abc"}],
                        "dependencies": [],
                    }
                ],
            }

    client = Client()
    queries = ("alien planet",)
    found, errors = platform_optimizer._parallel_neutral_shallow(queries, client)
    assert not errors
    assert found["alien planet"]

    # Candidate recall must not be version/loader-faceted.
    assert client.search_calls
    for _source, _query, kwargs in client.search_calls:
        assert "minecraft_version" not in kwargs
        assert "loader" not in kwargs
        assert kwargs["target_profile"] == "minecraft_mod"

    probes = (
        SimpleNamespace(
            adapter_id="probe:fabric:1.21.1",
            minecraft_version="1.21.1",
            loader="fabric",
        ),
        SimpleNamespace(
            adapter_id="probe:fabric:1.20.1",
            minecraft_version="1.20.1",
            loader="fabric",
        ),
    )
    matrix, matrix_errors = platform_optimizer._parallel_support_matrix(
        probes,
        queries,
        client,
    )
    assert not matrix_errors
    assert matrix["probe:fabric:1.21.1"]["alien planet"] == (
        "modrinth:compatible",
    )
    assert matrix["probe:fabric:1.20.1"]["alien planet"] == ()
    assert ("compatible", "1.21.1", "fabric") in client.inspect_calls


def test_mod_search_transport_failure_is_not_treated_as_no_mod_support() -> None:
    class BrokenClient:
        def search(self, source, query, **kwargs):
            raise TimeoutError("modrinth unavailable")

    client = BrokenClient()
    queries = ("worldgen",)
    found, errors = platform_optimizer._parallel_neutral_shallow(queries, client)
    assert found["worldgen"] == ()
    assert errors

    probe = SimpleNamespace(
        adapter_id="probe:fabric:1.21.1",
        minecraft_version="1.21.1",
        loader="fabric",
    )
    with pytest.raises(ValueError, match="source unavailable"):
        platform_optimizer._parallel_support_matrix((probe,), queries, client)


def test_structured_retry_uses_compact_repair_context_not_original_task() -> None:
    request = GenerationRequest(
        messages=(
            {"role": "system", "content": "S" * 20000},
            {"role": "user", "content": "U" * 30000},
        ),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )
    exc = SimpleNamespace(
        output='{"answer":7,"keep":"unchanged"}',
        errors=('$["answer"]: 7 is not of type "string"',),
    )
    repaired = _structured_repair_request(request, exc)
    joined = "\n".join(str(item.get("content", "")) for item in repaired.messages)

    assert len(joined) < 5000
    assert '"keep":"unchanged"' in joined
    assert "validation_errors" in joined
    assert "Change only the minimum" in joined
    assert repaired.media_paths == ()
    assert repaired.tools == ()
    assert repaired.tool_choice is None
