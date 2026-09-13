from __future__ import annotations

import json
from pathlib import Path

import yaml

from minecraft_mod_ai.planning_state_resolution import _generate_requirement_pages


class _PagedRequirementRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._pages = [
            {
                "requirements": [
                    {
                        "statement": "Gather resources and earn money.",
                        "semantic_capability": "economy",
                        "acceptance": ["Resources can be gathered and converted into money."],
                    },
                    {
                        "statement": "Trade resources and equipment.",
                        "semantic_capability": "trading",
                        "acceptance": ["A player can complete a trade."],
                    },
                    {
                        "statement": "Build a spacecraft from separate parts.",
                        "semantic_capability": "spacecraft_construction",
                        "acceptance": ["Separate spacecraft parts assemble into a usable craft."],
                    },
                    {
                        "statement": "Upgrade weapons, crew, and spacecraft performance through purchases or trades.",
                        "semantic_capability": "spacecraft_upgrade",
                        "acceptance": ["Purchased or traded upgrades change the relevant capability."],
                    },
                ]
            },
            {
                "requirements": [
                    {
                        "statement": "Launch the completed spacecraft into space and travel to other planets.",
                        "semantic_capability": "space_travel",
                        "acceptance": ["The player can leave the starting world and reach another planet."],
                    },
                    {
                        "statement": "Gather special minerals on other planets and fight aliens.",
                        "semantic_capability": "planet_exploration",
                        "acceptance": ["Planetary minerals and hostile aliens are both encountered in play."],
                    },
                    {
                        "statement": "Establish colonies on other planets.",
                        "semantic_capability": "colonization",
                        "acceptance": ["The player can create a persistent colony on another planet."],
                    },
                ]
            },
        ]

    def generate_tool_decision(self, role: str, messages: list[dict[str, str]], **kwargs: object) -> dict[str, object]:
        self.calls.append({"role": role, "messages": messages, **kwargs})
        return self._pages[len(self.calls) - 1]


def test_full_requirement_page_never_closes_semantic_frontier() -> None:
    router = _PagedRequirementRouter()
    messages = [
        {
            "role": "system",
            "content": "Compile independently testable player-visible requirements.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": {
                        "original_prompt": (
                            "우주모드 인데 자원파밍 돈모으기 거래 등으로 우주선을 부위마다 만들어서 "
                            "만들수있고 무기 선원 우주선 성능을 거래 구매 등으로 업그레이드 확장 할 수 "
                            "있고 그렇게해서 우주로 나갈수있고 우주로 나가면 다른행성의 특수 광물 "
                            "외게인과 싸움 식민지화등 여러가지가 가능한 모드"
                        )
                    }
                },
                ensure_ascii=False,
            ),
        },
    ]

    result = _generate_requirement_pages(router, messages, 100_000)

    assert len(router.calls) == 2
    assert len(result["requirements"]) == 7
    statements = " ".join(row["statement"] for row in result["requirements"])
    assert "spacecraft" in statements
    assert "other planets" in statements
    assert "aliens" in statements
    assert "colonies" in statements

    second_payload = json.loads(router.calls[1]["messages"][1]["content"])
    assert len(second_payload["already_compiled_requirements"]) == 4


def test_success_postcondition_primary_schema_matches_atomic_recovery_bound() -> None:
    template_path = (
        Path(__file__).resolve().parents[1]
        / "minecraft_mod_ai"
        / "templates"
        / "feature"
        / "behavior_contract"
        / "success_postconditions.yaml"
    )
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    properties = template["record_schema"]["properties"]

    assert properties["condition"]["maxLength"] == 256
    assert properties["observation"]["maxLength"] == 256
    rules = " ".join(template["rules"])
    assert "research summary" in rules
    assert "candidate descriptions" in rules
