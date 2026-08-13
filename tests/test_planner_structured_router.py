from __future__ import annotations

from minecraft_mod_ai.planner_structured_router import structured_planner_router


class _Router:
    def __init__(self) -> None:
        self.calls = []
        self.profile = "t4_local"

    def generate_text(self, role, messages, **kwargs):
        self.calls.append((role, messages, dict(kwargs)))
        return "{}"


def test_structured_planner_json_disables_tools() -> None:
    base = _Router()
    proxy = structured_planner_router(base)
    result = proxy.generate_text(
        "planner",
        [{"role": "user", "content": "return json"}],
        response_format="json",
    )
    assert result == "{}"
    assert base.calls[-1][2]["enable_tools"] is False


def test_non_planner_or_non_json_calls_keep_normal_policy() -> None:
    base = _Router()
    proxy = structured_planner_router(base)
    proxy.generate_text(
        "planner",
        [{"role": "user", "content": "design"}],
        response_format="text",
    )
    assert "enable_tools" not in base.calls[-1][2]

    proxy.generate_text(
        "coder",
        [{"role": "user", "content": "patch"}],
        response_format="json",
    )
    assert "enable_tools" not in base.calls[-1][2]


def test_proxy_is_idempotent_and_delegates_attributes() -> None:
    base = _Router()
    proxy = structured_planner_router(base)
    assert structured_planner_router(proxy) is proxy
    assert proxy.profile == "t4_local"
