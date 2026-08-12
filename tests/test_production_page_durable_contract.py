from __future__ import annotations

import json

from minecraft_mod_ai import complete_planner, work_graph
from minecraft_mod_ai.execution_efficiency_contract import install


class _PatchRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_text(
        self,
        role: str,
        messages,
        *,
        media_paths=(),
        response_format="text",
    ) -> str:
        self.calls.append(
            {
                "role": role,
                "messages": messages,
                "media_paths": media_paths,
                "response_format": response_format,
            }
        )
        request = json.loads(messages[-1]["content"])
        return json.dumps(
            {
                "target_fingerprint": request["target_fingerprint"],
                "set_fields": {"config": {}},
                "delete_fields": [],
            }
        )


def _module(module_id: str, *, config: bool = True) -> dict[str, object]:
    value: dict[str, object] = {
        "module_id": module_id,
        "kind": "item",
        "depends_on": [],
        "required_gates": [],
        "implements_deliverables": [module_id],
    }
    if config:
        value["config"] = {}
    return value


def test_invalid_production_item_patches_only_missing_field_and_keeps_sibling(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    install(complete_planner_module=complete_planner, work_graph_module=work_graph)

    page_calls = 0

    def page_generator(
        router,
        *,
        system_prompt,
        request,
        media_paths,
        expected_contracts,
        stage,
    ):
        nonlocal page_calls
        page_calls += 1
        return {
            "modules": [
                _module("good"),
                _module("needs_config", config=False),
            ],
            "assets": [],
            "audio": [],
            "acceptance_tests": ["good and needs_config exist"],
            "completed_deliverables": ["good", "needs_config"],
            "complete": True,
            "next_cursor": "",
        }

    monkeypatch.setattr(
        complete_planner,
        "_generate_json_page_with_repair",
        page_generator,
    )

    router = _PatchRouter()
    planner = object.__new__(complete_planner.CompleteGameDesignPlanner)
    planner.router = router
    batch = complete_planner._ProductionBatch(
        batch_id="durable_items",
        scope="two items",
        depends_on_batches=(),
        deliverables=("good", "needs_config"),
        exports=(),
    )
    parts = complete_planner._ProductionParts([], [], [], [])

    planner._expand_one_production_batch(
        batch=batch,
        parts=parts,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )

    assert page_calls == 1
    assert [item.module_id for item in parts.modules] == ["good", "needs_config"]
    assert len(router.calls) == 1
    patch_system = str(router.calls[0]["messages"][0]["content"])
    patch_user = json.loads(router.calls[0]["messages"][1]["content"])
    assert "DO NOT rewrite the whole object" in patch_system
    assert "config" in patch_user["validation_error"]
    assert patch_user["current_value"]["module_id"] == "needs_config"

    # Rebuild fresh in-memory catalogs/parts as a restarted planner would. The exact
    # production page and the resolved field patch come from disk; no page generation
    # or patch LLM call repeats.
    parts_replayed = complete_planner._ProductionParts([], [], [], [])
    router_replayed = _PatchRouter()
    planner_replayed = object.__new__(complete_planner.CompleteGameDesignPlanner)
    planner_replayed.router = router_replayed
    planner_replayed._expand_one_production_batch(
        batch=batch,
        parts=parts_replayed,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )

    assert page_calls == 1
    assert router_replayed.calls == []
    assert [item.module_id for item in parts_replayed.modules] == [
        "good",
        "needs_config",
    ]


def test_duplicate_production_item_id_is_patched_not_dropped(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_PLANNER_CHECKPOINT_DIR", str(tmp_path))
    install(complete_planner_module=complete_planner, work_graph_module=work_graph)

    class _RenameRouter(_PatchRouter):
        def generate_text(self, role, messages, *, media_paths=(), response_format="text"):
            self.calls.append({"role": role, "messages": messages})
            request = json.loads(messages[-1]["content"])
            return json.dumps(
                {
                    "target_fingerprint": request["target_fingerprint"],
                    "set_fields": {"module_id": "duplicate_fixed"},
                    "delete_fields": [],
                }
            )

    def page_generator(*args, **kwargs):
        return {
            "modules": [_module("duplicate"), _module("duplicate")],
            "assets": [],
            "audio": [],
            "acceptance_tests": ["both implementations exist"],
            "completed_deliverables": ["d1", "d2"],
            "complete": True,
            "next_cursor": "",
        }

    monkeypatch.setattr(complete_planner, "_generate_json_page_with_repair", page_generator)
    planner = object.__new__(complete_planner.CompleteGameDesignPlanner)
    planner.router = _RenameRouter()
    parts = complete_planner._ProductionParts([], [], [], [])
    planner._expand_one_production_batch(
        batch=complete_planner._ProductionBatch(
            batch_id="duplicate_ids",
            scope="two items",
            depends_on_batches=(),
            deliverables=("d1", "d2"),
            exports=(),
        ),
        parts=parts,
        module_catalog=complete_planner._ModuleCatalog(),
        asset_catalog=complete_planner._ModuleCatalog(),
        audio_catalog=complete_planner._ModuleCatalog(),
        test_catalog=set(),
        dependency_exports={},
        planning_context={},
        planning_receipt={},
        media_paths=(),
    )

    assert [item.module_id for item in parts.modules] == ["duplicate", "duplicate_fixed"]
