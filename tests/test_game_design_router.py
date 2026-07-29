import json

from minecraft_mod_ai.game_design import GameDesignPlanner


class FakeRouter:
    def generate_text(self, role, messages, **kwargs):
        assert role == "planner"
        return json.dumps(
            {
                "game_design": {
                    "title": "Moon Forge",
                    "pitch": "Craft moon relics and fight in an arena.",
                    "core_loop": ["mine", "craft", "fight"],
                    "progression": ["ore", "relic", "boss"],
                    "combat": {"player_verbs": ["strike"], "enemy_roles": ["boss"]},
                    "world": {"regions": [{"id": "moon_arena", "purpose": "boss", "links": []}]},
                    "modules": [
                        {"plugin_id": "fabric-core", "status": "implemented", "reason": "available"},
                        {"plugin_id": "quest-system", "status": "blocked", "reason": "not implemented"},
                    ],
                    "assets": [{"id": "moon_crystal", "kind": "item", "brief": "blue crystal"}],
                    "acceptance_tests": ["item is registered"],
                },
                "build_slice": {
                    "mod_id": "moon_forge",
                    "mod_name": "Moon Forge",
                    "package_name": "ai.minecraft.generated.moon_forge",
                    "summary": "Moon content",
                    "contents": [
                        {
                            "content_id": "moon_crystal",
                            "kind": "item",
                            "display_name_en": "Moon Crystal",
                            "display_name_ko": "달 결정",
                            "color": "#89dceb",
                            "recipe": True,
                        }
                    ],
                    "deferred_capabilities": ["quest_system"],
                },
            },
            ensure_ascii=False,
        )


def test_multimodal_design_keeps_blocked_modules_visible() -> None:
    design, proposal = GameDesignPlanner(FakeRouter()).plan(
        "달 결정 아이템과 퀘스트를 만들어줘"
    )
    assert any(module["status"] == "blocked" for module in design["modules"])
    assert proposal.spec.mod_id == "moon_forge"
    assert proposal.approval_hash == proposal.calculate_hash()
