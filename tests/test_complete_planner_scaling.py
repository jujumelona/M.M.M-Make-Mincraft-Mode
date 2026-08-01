from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _ProductionBatch,
    _remove_bootstrap_duplicates,
    _merge_world_fragments,
)
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.spec import ContentKind, ContentSpec, SpecValidationError


def _module(module_id: str) -> dict[str, object]:
    return {
        "module_id": module_id,
        "kind": "custom_java",
        "config": {},
        "depends_on": [],
        "required_gates": [],
    }


class _LongPaginationRouter:
    def __init__(self, page_count: int) -> None:
        self.page_count = page_count
        self.requests: list[dict[str, object]] = []
        self.request_sizes: list[int] = []
        self.media_paths: list[tuple[str, ...]] = []
        self.expected_cursor = ""

    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        user_content = messages[-1]["content"]
        request = json.loads(user_content)
        assert request["cursor"] == self.expected_cursor
        self.requests.append(request)
        self.request_sizes.append(len(user_content.encode("utf-8")))
        self.media_paths.append(tuple(str(path) for path in kwargs["media_paths"]))

        index = len(self.requests) - 1
        complete = index == self.page_count - 1
        next_cursor = (
            ""
            if complete
            else f"opaque/{index + 1:08d}?token=unchanged-width"
        )
        self.expected_cursor = next_cursor
        return json.dumps(
            {
                "modules": [_module(f"feature_{index:08d}")],
                "complete": complete,
                "next_cursor": next_cursor,
            }
        )


def test_module_pagination_request_stays_bounded_across_many_pages() -> None:
    router = _LongPaginationRouter(page_count=600)
    planner = CompleteGameDesignPlanner(router)
    huge_evidence_marker = "must-not-be-resent-" * 20_000

    modules = planner._expand_batches(
        prompt="Build every system in this self-contained batch.",
        game_design={
            "title": "Unbounded production graph",
            "pitch": "Compile the complete requested graph.",
            "modules": [{"plugin_id": "custom", "reason": "requested"}],
            "_technical_evidence": {
                "schema_version": "test/evidence-v1",
                "untrusted_excerpt": huge_evidence_marker,
            },
        },
        batches=[
            {
                "batch_id": "all_requested_systems",
                "scope": (
                    "Implement all 600 requested independent systems and their "
                    "observable validation hooks."
                ),
                "depends_on_batches": [],
            }
        ],
        media_paths=("reference.png",),
    )

    assert len(modules) == 600
    assert len({module.module_id for module in modules}) == 600
    assert "planning_context" in router.requests[0]
    assert all(
        "planning_context" not in request
        for request in router.requests[1:]
    )
    assert all("known_module_ids" not in request for request in router.requests)
    assert all(
        huge_evidence_marker not in json.dumps(request)
        for request in router.requests
    )

    for index, request in enumerate(router.requests):
        catalog = request["known_module_catalog"]
        assert catalog["count"] == index
        assert len(catalog["sha256"]) == 64
        assert len(catalog["recent_ids"]) <= 32
        assert catalog["recent_limit"] == 32
    assert len(set(
        request["known_module_catalog"]["sha256"]
        for request in router.requests
    )) == 600

    steady_state_sizes = router.request_sizes[64:]
    assert max(steady_state_sizes) - min(steady_state_sizes) <= 2
    assert max(router.request_sizes) < 5_000
    assert router.media_paths[0] == ("reference.png",)
    assert all(not paths for paths in router.media_paths[1:])


class _ResponseRouter:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, object]] = []
        self.media_paths: list[tuple[str, ...]] = []

    def generate_text(self, role, messages, **kwargs):
        self.requests.append(json.loads(messages[-1]["content"]))
        self.media_paths.append(tuple(
            str(path) for path in kwargs["media_paths"]
        ))
        return json.dumps(next(self.responses))


def test_module_pagination_rejects_duplicate_ids_within_page() -> None:
    planner = CompleteGameDesignPlanner(
        _ResponseRouter(
            [
                {
                    "modules": [_module("same_id"), _module("same_id")],
                    "complete": True,
                    "next_cursor": "",
                }
            ]
        )
    )

    with pytest.raises(SpecValidationError, match="duplicate module ID"):
        planner._expand_batches(
            prompt="Duplicate test",
            game_design={"title": "Duplicate test"},
            batches=[
                {
                    "batch_id": "one",
                    "scope": "one",
                    "depends_on_batches": [],
                }
            ],
            media_paths=(),
        )


def test_module_pagination_rejects_duplicate_ids_across_batches() -> None:
    planner = CompleteGameDesignPlanner(
        _ResponseRouter(
            [
                {
                    "modules": [_module("shared_id")],
                    "complete": True,
                    "next_cursor": "",
                },
                {
                    "modules": [_module("shared_id")],
                    "complete": True,
                    "next_cursor": "",
                },
            ]
        )
    )

    with pytest.raises(SpecValidationError, match="duplicate module ID"):
        planner._expand_batches(
            prompt="Global duplicate test",
            game_design={"title": "Global duplicate test"},
            batches=[
                {
                    "batch_id": "one",
                    "scope": "first",
                    "depends_on_batches": [],
                },
                {
                    "batch_id": "two",
                    "scope": "second",
                    "depends_on_batches": ["one"],
                },
            ],
            media_paths=(),
        )


def test_full_planning_context_and_media_are_sent_once_across_batches() -> None:
    router = _ResponseRouter(
        [
            {
                "modules": [_module(f"module_{index}")],
                "complete": True,
                "next_cursor": "",
            }
            for index in range(3)
        ]
    )
    planner = CompleteGameDesignPlanner(router)

    modules = planner._expand_batches(
        prompt="Build the complete requested design.",
        game_design={
            "title": "Many batches",
            "modules": [
                {"plugin_id": f"feature_{index}", "reason": "requested"}
                for index in range(1_000)
            ],
            "_technical_evidence": {"excerpt": "untrusted-" * 10_000},
        },
        batches=[
            {
                "batch_id": f"batch_{index}",
                "scope": f"Self-contained implementation scope {index}",
                "depends_on_batches": (
                    [] if index == 0 else [f"batch_{index - 1}"]
                ),
            }
            for index in range(3)
        ],
        media_paths=("large-reference.png",),
    )

    assert len(modules) == 3
    assert "planning_context" in router.requests[0]
    assert all(
        "planning_context" not in request
        for request in router.requests[1:]
    )
    assert len({
        request["planning_context_receipt"]["sha256"]
        for request in router.requests
    }) == 1
    assert router.media_paths == [
        ("large-reference.png",),
        (),
        (),
    ]


def test_scalable_production_batches_track_explicit_remaining_work() -> None:
    router = _ResponseRouter(
        [
            {
                "modules": [_module("core_module")],
                "world_ir_fragment": None,
                "assets": [],
                "audio": [],
                "acceptance_tests": [],
                "completed_deliverables": ["core"],
                "complete": False,
                "next_cursor": "core-page-2",
            },
            {
                "modules": [],
                "world_ir_fragment": {
                    "regions": [{"id": "capital", "purpose": "start"}],
                },
                "assets": [],
                "audio": [],
                "acceptance_tests": ["capital loads"],
                "completed_deliverables": ["capital_world"],
                "complete": True,
                "next_cursor": "",
            },
            {
                "modules": [
                    {
                        **_module("travel_module"),
                        "depends_on": ["core_module"],
                    }
                ],
                "world_ir_fragment": {
                    "regions": [{"id": "outpost", "purpose": "destination"}],
                    "routes": [{"from": "capital", "to": "outpost"}],
                },
                "assets": [],
                "audio": [],
                "acceptance_tests": ["travel reaches outpost"],
                "completed_deliverables": ["travel"],
                "complete": True,
                "next_cursor": "",
            },
        ]
    )
    planner = CompleteGameDesignPlanner(router)
    parts = planner._expand_production_batches(
        batches=(
            _ProductionBatch(
                "core",
                "Implement core and capital.",
                (),
                ("core", "capital_world"),
                ("core_module",),
            ),
            _ProductionBatch(
                "travel",
                "Implement travel using the core export.",
                ("core",),
                ("travel",),
                ("travel_module",),
            ),
        ),
        prompt="Create the requested world and travel system.",
        game_design={"title": "Scalable"},
        media_paths=("reference.png",),
    )

    assert [item.module_id for item in parts.modules] == [
        "core_module",
        "travel_module",
    ]
    assert router.requests[0]["remaining_deliverables"] == [
        "core",
        "capital_world",
    ]
    assert router.requests[1]["remaining_deliverables"] == ["capital_world"]
    assert router.requests[2]["dependency_exports"] == {
        "core": ["core_module"]
    }
    merged = _merge_world_fragments(parts.world_fragments)
    assert merged is not None
    assert [item["id"] for item in merged["regions"]] == [
        "capital",
        "outpost",
    ]
    assert merged["routes"] == [{"from": "capital", "to": "outpost"}]


def test_production_outline_paginates_without_repeating_full_evidence() -> None:
    router = _ResponseRouter(
        [
            {
                "production_batches": [
                    {
                        "batch_id": "second",
                        "scope": "Second implementation scope.",
                        "depends_on_batches": ["first"],
                        "deliverables": ["second work"],
                        "exports": ["second_module"],
                    }
                ],
                "complete": True,
                "next_cursor": "",
            }
        ]
    )
    planner = CompleteGameDesignPlanner(router)
    batches = planner._collect_production_batches(
        first_page={
            "production_batches": [
                {
                    "batch_id": "first",
                    "scope": "First implementation scope.",
                    "depends_on_batches": [],
                    "deliverables": ["first work"],
                    "exports": ["first_module"],
                }
            ],
            "complete": False,
            "next_cursor": "outline-page-2",
        },
        prompt="Build a large project.",
        game_design={
            "title": "Large",
            "_technical_evidence": {"huge": "not-forwarded" * 1000},
        },
        media_paths=("reference.png",),
    )

    assert [item.batch_id for item in batches] == ["first", "second"]
    assert router.requests[0]["cursor"] == "outline-page-2"
    assert "_technical_evidence" not in router.requests[0]["planning_context"][
        "game_design"
    ]
    assert router.media_paths == [()]


def _bootstrap_base():
    content = ContentSpec(
        content_id="bootstrap_relic",
        kind=ContentKind.ITEM,
        display_name_en="Bootstrap Relic",
        display_name_ko="Bootstrap Relic KO",
        color="#123456",
        recipe=True,
    )
    return SimpleNamespace(
        spec=SimpleNamespace(contents=(content,))
    )


def test_equivalent_bootstrap_duplicate_remains_backward_compatible() -> None:
    modules = _remove_bootstrap_duplicates(
        (
            ProductionModule(
                module_id="bootstrap_relic",
                kind="item",
                config={
                    "display_name_en": "Bootstrap Relic",
                    "display_name_ko": "Bootstrap Relic KO",
                    "color": "#123456",
                    "recipe": True,
                },
                required_gates=("registry", "resource", "recipe"),
            ),
        ),
        _bootstrap_base(),
    )

    assert len(modules) == 1
    assert modules[0].module_id == "bootstrap_integration"
    assert modules[0].config == {
        "uses_base_content": ["bootstrap_relic"]
    }


@pytest.mark.parametrize(
    "module",
    (
        ProductionModule(
            module_id="bootstrap_relic",
            kind="item",
            config={"attack_damage": 12},
        ),
        ProductionModule(
            module_id="bootstrap_relic",
            kind="item",
            depends_on=("progression_core",),
        ),
        ProductionModule(
            module_id="bootstrap_relic",
            kind="item",
            required_gates=("multiplayer runtime proof",),
        ),
    ),
)
def test_richer_bootstrap_duplicate_routes_to_custom_extension(
    module: ProductionModule,
) -> None:
    modules = _remove_bootstrap_duplicates((module,), _bootstrap_base())

    assert len(modules) == 1
    extension = modules[0]
    assert extension.module_id == module.module_id
    assert extension.kind == "custom_java"
    assert extension.config == {
        **module.config,
        "requested_kind": "item",
        "extends_bootstrap": "bootstrap_relic",
    }
    assert extension.depends_on == module.depends_on
    assert extension.required_gates == module.required_gates


def test_bootstrap_duplicate_kind_mismatch_still_fails_closed() -> None:
    with pytest.raises(
        SpecValidationError,
        match="collides with bootstrap item",
    ):
        _remove_bootstrap_duplicates(
            (
                ProductionModule(
                    module_id="bootstrap_relic",
                    kind="block",
                ),
            ),
            _bootstrap_base(),
        )
