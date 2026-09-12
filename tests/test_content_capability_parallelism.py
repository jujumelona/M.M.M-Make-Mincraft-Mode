from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from minecraft_mod_ai import content_design_graph as content_graph


class _Registry:
    @staticmethod
    def role(profile, role):
        del profile, role
        return SimpleNamespace(
            exclusive_gpu=True,
            provider="local",
            adapter="llama_cpp",
        )


class _Router:
    profile = "test"
    registry = _Registry()


def test_content_capabilities_use_native_slots_and_preserve_entity_order(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "2")
    router = _Router()
    lock = threading.Lock()
    active = 0
    maximum = 0
    checkpoint_active = 0
    checkpoint_maximum = 0
    checkpoint_bindings: list[str] = []

    entity_contexts = {
        f"entity_{index}": {"entity": {"entity_id": f"entity_{index}"}}
        for index in range(6)
    }

    def checkpoint(binding, value):
        nonlocal checkpoint_active, checkpoint_maximum
        with lock:
            checkpoint_active += 1
            checkpoint_maximum = max(checkpoint_maximum, checkpoint_active)
        time.sleep(0.005)
        checkpoint_bindings.append(binding)
        assert value["fact_type"] == "item_exists"
        with lock:
            checkpoint_active -= 1

    def fake_run_record_template(
        model_router,
        identifier,
        *,
        context,
        allowed_refs,
        progress,
        checkpoint,
    ):
        nonlocal active, maximum
        assert model_router is router
        assert identifier == "design/content_capability"
        assert allowed_refs == ()
        assert isinstance(progress, dict)
        entity_id = context["entity"]["entity_id"]
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        if checkpoint is not None:
            checkpoint(f"capability:{entity_id}", {"fact_type": "item_exists"})
        with lock:
            active -= 1
        return {
            "records": [{"fact_type": "item_exists", "entity_id": entity_id}],
            "reason": "",
            "evidence_refs": [],
        }

    monkeypatch.setattr(content_graph, "run_record_template", fake_run_record_template)

    result = content_graph._resolve_content_capabilities(
        router,
        entity_contexts,
        progress={},
        checkpoint=checkpoint,
    )

    assert list(result) == list(entity_contexts)
    assert [row["entity_id"] for row in result.values()] == list(entity_contexts)
    assert maximum == 2
    assert checkpoint_maximum == 1
    assert sorted(checkpoint_bindings) == sorted(
        f"capability:{entity_id}" for entity_id in entity_contexts
    )
