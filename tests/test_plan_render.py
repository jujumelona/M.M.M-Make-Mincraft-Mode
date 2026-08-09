from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.plan_render import render_complete_plan


def test_korean_plan_is_conversational_and_hides_internal_protocol() -> None:
    text = render_complete_plan(
        requested_prompt="농사와 요리 모드를 만들어줘",
        game_design={
            "title": "계절의 식탁",
            "pitch": "농작물을 길러 요리를 완성한다.",
            "core_loop": ["씨앗을 찾는다", "재배한다", "요리한다"],
            "progression": ["새 조리법을 연다"],
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [
                {
                    "plugin_id": "cooking",
                    "reason": "재배와 조리를 연결한다",
                }
            ],
        },
        modules=(
            ProductionModule("tomato_crop", "crop"),
            ProductionModule("tomato_stew", "food"),
        ),
        acceptance_tests=("토마토를 재배해 수프를 만든다",),
    )

    assert "이 방향으로 만들까요?" in text
    assert "재배와 조리를 연결한다" in text
    assert "작물과 재배" in text
    assert "음식과 요리" in text
    assert "cooking" not in text
    assert "crop 1" not in text
    assert "맵" not in text
    assert "boss" not in text.lower()
    assert "보스" not in text
    assert "승인" not in text
    assert "sha256" not in text.lower()
    assert "rag" not in text.lower()
    assert "mcp" not in text.lower()


def test_plan_does_not_suggest_unrequested_map_or_encounter_shapes() -> None:
    text = render_complete_plan(
        requested_prompt="Create a tiny decorative lantern mod.",
        game_design={
            "title": "Warm Lanterns",
            "pitch": "Decorative lights with warm color variants.",
            "core_loop": ["Craft a lantern", "Place and recolor it"],
            "progression": [],
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [],
        },
        modules=(ProductionModule("warm_lantern", "block"),),
        acceptance_tests=("Craft and place every lantern variant.",),
    )

    assert "blocks" in text
    assert "boss" not in text.lower()
    assert "map" not in text.lower()
    assert "arena" not in text.lower()


def test_internal_research_shards_do_not_advertise_mod_integration() -> None:
    text = render_complete_plan(
        requested_prompt="Create a tiny decorative lantern mod.",
        game_design={
            "title": "Warm Lanterns",
            "pitch": "Decorative lights only.",
            "core_loop": ["Craft and place a lantern"],
            "progression": [],
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [],
        },
        modules=(
            ProductionModule("warm_lantern", "block"),
            ProductionModule(
                "mmm_research_ledger",
                "integration",
                {"integration_type": "mmm_research_shard"},
            ),
        ),
        acceptance_tests=("The lantern is placeable.",),
    )

    assert "blocks" in text
    assert "integration with other mods" not in text


def test_plan_surfaces_requested_art_direction_without_inventing_gameplay() -> None:
    text = render_complete_plan(
        requested_prompt="Create a moon relic visual mod.",
        game_design={
            "title": "Moon Relics",
            "pitch": "Lunar artifacts with a clear readable silhouette.",
            "core_loop": ["Craft and place relics"],
            "progression": [],
            "art_direction": {
                "visual_tone": "cool moonlit blue",
                "texture_guidance": ["16x16 clean contrast"],
                "model_animation_guidance": ["subtle glow pulse"],
            },
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [],
        },
        modules=(),
        acceptance_tests=("Visual direction is visible in the plan.",),
    )

    assert "Visual direction" in text
    assert "cool moonlit blue" in text
    assert "Textures: 16x16 clean contrast" in text
    assert "3D and animation: subtle glow pulse" in text
    assert "boss" not in text.lower()


def test_large_plan_preview_is_bounded_without_dropping_stored_scope() -> None:
    text = render_complete_plan(
        requested_prompt="Create a very large content mod.",
        game_design={
            "title": "Large Archive",
            "pitch": "A large but paged production plan.",
            "core_loop": [f"Loop step {index}" for index in range(100)],
            "progression": [],
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [],
        },
        modules=tuple(
            ProductionModule(f"module_{index:05d}", "item")
            for index in range(100)
        ),
        acceptance_tests=tuple(
            f"Acceptance test {index}" for index in range(100)
        ),
    )

    assert "76 more entries remain in the stored plan" in text
    assert "Acceptance test 99" not in text
    assert len(text) < 20_000


def test_plan_explains_quality_contract_in_plain_language() -> None:
    text = render_complete_plan(
        requested_prompt="작은 나침반 모드를 만들어줘",
        game_design={
            "title": "날씨 나침반",
            "pitch": "현재 날씨를 보여 준다.",
            "core_loop": ["나침반을 확인한다"],
            "progression": [],
            "mod_context": {
                "vanilla_integration": [],
                "compatibility_targets": [],
            },
            "modules": [],
            "_production_contract": {
                "catalog_stats": {
                    "requirements": 7,
                    "acceptance_tests": 12,
                },
                "quality_dimension_catalog": [
                    {"title": "Requirement correctness"},
                    {"title": "Runtime behavior"},
                ],
            },
        },
        modules=(ProductionModule("weather_compass", "item"),),
        acceptance_tests=("게임에서 날씨를 확인한다",),
    )

    assert "완성 기준" in text
    assert "요구사항 7개" in text
    assert "독립 검증 증거" in text
    assert "sha256" not in text.casefold()
