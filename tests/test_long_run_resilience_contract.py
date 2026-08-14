from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import long_run_resilience_contract as resilience


def _valid_note(*, sufficient: bool = False) -> str:
    return json.dumps(
        {
            "research_note": {
                "domain_id": "mk_project",
                "claims": [{"claim": "kept", "evidence_refs": ["page:1"]}],
                "gaps": [] if sufficient else ["needs nothing new"],
                "next_queries": [],
                "sufficient": sufficient,
            }
        }
    )


def _document_messages(*, prior=None, sentinel: str = "TAIL-SENTINEL"):
    payload = {
        "authoritative_request": f"build the mod {sentinel}",
        "domain": {"domain_id": "mk_project"},
        "deterministic_evidence": {
            "evidence_document": {
                "document_sha256": "sha256:doc-one",
                "page_count": 8,
            },
            "page_notes": [
                {"page_ref": "page:1", "claims": [{"claim": sentinel}]},
                {"page_ref": "page:8", "claims": [{"claim": "last-page"}]},
            ],
        },
    }
    if prior is not None:
        payload["previous_reflection"] = prior
    return [
        {"role": "system", "content": "synthesize all supplied page notes"},
        {"role": "user", "content": json.dumps(payload, sort_keys=True)},
    ]


def _page_messages(sentinel: str = "PAGE-TAIL-SENTINEL"):
    return [
        {"role": "system", "content": "read the complete page"},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "evidence_document": {
                        "document_sha256": "sha256:page-doc",
                    },
                    "evidence_page": {
                        "page_ref": "page:7",
                        "content": f"full evidence {sentinel}",
                    },
                },
                sort_keys=True,
            ),
        },
    ]


def test_document_synthesis_disables_duplicate_tools_and_reaches_host_fixed_point(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MMM_RESEARCH_CHECKPOINT_ROOT", str(tmp_path))
    resilience._SYNTHESIS_RESULTS.clear()
    calls = []

    class Router:
        profile = "test"
        registry = None

        def generate_text(self, role, messages, *args, **kwargs):
            calls.append((role, messages, args, dict(kwargs)))
            return _valid_note(sufficient=False)

    module = SimpleNamespace(ModelRouter=Router)
    resilience._install_research_generation_resilience(module)
    router = Router()

    first = router.generate_text(
        "planner",
        _document_messages(),
        tool_stage="research",
        enable_tools=True,
        response_format="json",
    )
    second = router.generate_text(
        "planner",
        _document_messages(prior=json.loads(first)["research_note"]),
        tool_stage="research",
        enable_tools=True,
        response_format="json",
    )

    assert first == second
    assert len(calls) == 1
    assert calls[0][3]["enable_tools"] is False
    assert "TAIL-SENTINEL" in calls[0][1][1]["content"]


def test_successful_research_page_generation_is_durably_reused(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MMM_RESEARCH_CHECKPOINT_ROOT", str(tmp_path))
    calls = []

    class Router:
        profile = "test"
        registry = None

        def generate_text(self, role, messages, *args, **kwargs):
            calls.append(messages)
            return _valid_note(sufficient=True)

    module = SimpleNamespace(ModelRouter=Router)
    resilience._install_research_generation_resilience(module)
    router = Router()
    messages = _page_messages()

    first = router.generate_text(
        "planner",
        messages,
        tool_stage="research",
        enable_tools=False,
        response_format="json",
    )
    second = router.generate_text(
        "planner",
        messages,
        tool_stage="research",
        enable_tools=False,
        response_format="json",
    )

    assert first == second
    assert len(calls) == 1
    assert "PAGE-TAIL-SENTINEL" in calls[0][1]["content"]
    assert list(tmp_path.rglob("*.json"))


def test_corrupt_research_checkpoint_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_RESEARCH_CHECKPOINT_ROOT", str(tmp_path))
    calls = []

    class Router:
        profile = "test-corrupt"
        registry = None

        def generate_text(self, role, messages, *args, **kwargs):
            calls.append(messages)
            return _valid_note(sufficient=True)

    module = SimpleNamespace(ModelRouter=Router)
    resilience._install_research_generation_resilience(module)
    router = Router()
    messages = _page_messages("CORRUPT-TAIL")
    kwargs = {
        "tool_stage": "research",
        "enable_tools": False,
        "response_format": "json",
    }
    key = resilience._research_request_key(router, "planner", messages, (), kwargs)
    path = resilience._checkpoint_path(key)
    path.write_text('{"schema_version":"broken"}', encoding="utf-8")

    raw = router.generate_text("planner", messages, **kwargs)

    assert json.loads(raw)["research_note"]["sufficient"] is True
    assert len(calls) == 1
    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert repaired["schema_version"] == resilience._CACHE_SCHEMA


class _DeadProcess:
    def poll(self):
        return 17


class _LiveProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _autotune(process, *, url="http://127.0.0.1:8910/v1"):
    return SimpleNamespace(
        _AUTOTUNE_LOCK=resilience.threading.RLock(),
        _MANAGED_PROCESS=process,
        _MANAGED_KEY="model-key",
        _MANAGED_URL=url,
        _ATTEMPTED_KEYS={"model-key"},
    )


def test_dead_managed_llama_server_is_rearmed_and_attempt_guard_cleared(monkeypatch):
    autotune = _autotune(_DeadProcess())
    monkeypatch.setenv("LLAMA_SERVER_URL", autotune._MANAGED_URL)

    assert resilience._rearm_managed_server(autotune, force=False) is True
    assert autotune._MANAGED_PROCESS is None
    assert autotune._MANAGED_KEY is None
    assert autotune._MANAGED_URL is None
    assert "model-key" not in autotune._ATTEMPTED_KEYS
    assert "LLAMA_SERVER_URL" not in os.environ


def test_managed_transport_failure_replays_exact_request_once(monkeypatch):
    process = _LiveProcess()
    autotune = _autotune(process)
    monkeypatch.setenv("LLAMA_SERVER_URL", autotune._MANAGED_URL)
    seen = []

    class Adapter:
        def generate(self, request):
            seen.append(request)
            if len(seen) == 1:
                raise RuntimeError(
                    "ModelBackendError: [Errno 111] Connection refused"
                )
            return "recovered"

    module = SimpleNamespace(LlamaCppAdapter=Adapter)
    resilience._install_managed_backend_recovery(module, autotune)
    request = object()

    assert Adapter().generate(request) == "recovered"
    assert seen == [request, request]
    assert process.terminated is True
    assert "model-key" not in autotune._ATTEMPTED_KEYS


def test_explicit_external_server_is_never_restarted(monkeypatch):
    process = _LiveProcess()
    autotune = _autotune(process)
    monkeypatch.setenv("LLAMA_SERVER_URL", "http://external.example/v1")
    calls = 0

    class Adapter:
        def generate(self, request):
            nonlocal calls
            calls += 1
            raise RuntimeError(
                "ModelBackendError: [Errno 111] Connection refused"
            )

    module = SimpleNamespace(LlamaCppAdapter=Adapter)
    resilience._install_managed_backend_recovery(module, autotune)

    with pytest.raises(RuntimeError, match="Connection refused"):
        Adapter().generate(object())

    assert calls == 1
    assert process.terminated is False
    assert autotune._MANAGED_KEY == "model-key"
