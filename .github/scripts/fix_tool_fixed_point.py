from __future__ import annotations

from pathlib import Path

ROUTER = Path("minecraft_mod_ai/model_router.py")
TEST = Path("tests/test_agent_tool_calling.py")


def main() -> None:
    source = ROUTER.read_text(encoding="utf-8")

    old_doc = '''        """Gather tool evidence until the model itself returns a final answer.\n\n        No host-owned tool-round or tool-call ceiling exists. The only loop guard\n        is semantic: two consecutive identical tool-call/result exchanges prove\n        an exact no-progress fixed point.\n        """'''
    new_doc = '''        """Gather tool evidence until the model itself returns a final answer.\n\n        No host-owned tool-round or tool-call ceiling exists. The semantic loop\n        guard detects two consecutive identical tool-call/result exchanges. Exact\n        convergence closes tool use and forces a final synthesis from accumulated\n        observations instead of misclassifying convergence as model misconfiguration.\n        """'''
    if old_doc not in source:
        raise SystemExit("model_router tool-loop docstring anchor not found")
    source = source.replace(old_doc, new_doc, 1)

    old_block = '''            if exchange_state == previous_exchange_state:\n                raise ModelConfigurationError(\n                    "Agent reached an exact no-progress tool fixed point: identical "\n                    "tool calls produced identical observations on consecutive turns."\n                )\n            previous_exchange_state = exchange_state\n            round_index += 1'''
    new_block = '''            if exchange_state == previous_exchange_state:\n                final_messages = [\n                    *messages,\n                    {\n                        "role": "system",\n                        "content": (\n                            "Tool use has converged: the immediately preceding tool "\n                            "exchange repeated an identical call and identical "\n                            "observation. Do not call any more tools. Return the final "\n                            "answer now using only the evidence already present in this "\n                            "conversation. Preserve the requested response format and "\n                            "do not mention this convergence instruction."\n                        ),\n                    },\n                ]\n                final_request = GenerationRequest(\n                    messages=final_messages,\n                    media_paths=(),\n                    response_format=request.response_format,\n                    tools=(),\n                    tool_choice=None,\n                    parallel_tool_calls=False,\n                )\n                final_turn = adapter.generate_turn(final_request)\n                if final_turn.tool_calls:\n                    raise ModelConfigurationError(\n                        "Agent emitted tool calls after tools were disabled at an exact "\n                        "no-progress fixed point."\n                    )\n                final_content = final_turn.content.strip()\n                if not final_content:\n                    raise ModelConfigurationError(\n                        "Agent returned an empty final response after exact tool "\n                        "fixed-point convergence."\n                    )\n                return final_content\n            previous_exchange_state = exchange_state\n            round_index += 1'''
    if old_block not in source:
        raise SystemExit("model_router fixed-point block anchor not found")
    source = source.replace(old_block, new_block, 1)
    ROUTER.write_text(source, encoding="utf-8")

    test_source = TEST.read_text(encoding="utf-8")
    old_test = '''def test_agent_stops_on_consecutive_exact_tool_fixed_point(monkeypatch) -> None:\n    class LoopAdapter:\n        def generate_turn(self, request):\n            return GenerationResponse(\n                tool_calls=(\n                    ToolCall(\n                        id="call",\n                        name="search_code_rag",\n                        arguments={"query": "same"},\n                        raw_arguments='{\"query\":\"same\"}',\n                    ),\n                )\n            )\n\n    adapter = LoopAdapter()\n    runtime = _ToolRuntime()\n    monkeypatch.setattr(\n        ModelRouter,\n        "_new_text_adapter",\n        staticmethod(lambda config, *, role: adapter),\n    )\n    router = ModelRouter(\n        profile="test",\n        registry=_Registry(),\n        agent_tool_runtime_factory=lambda **_: runtime,\n    )\n    with pytest.raises(ModelConfigurationError, match="no-progress tool fixed point"):\n        router.generate_text("coder", [{"role": "user", "content": "research"}])\n    assert len(runtime.calls) == 2\n'''
    new_test = '''def test_agent_synthesizes_final_answer_on_consecutive_exact_tool_fixed_point(\n    monkeypatch,\n) -> None:\n    class LoopAdapter:\n        def __init__(self) -> None:\n            self.requests = []\n\n        def generate_turn(self, request):\n            self.requests.append(request)\n            if len(self.requests) <= 2:\n                return GenerationResponse(\n                    tool_calls=(\n                        ToolCall(\n                            id=f"call_{len(self.requests)}",\n                            name="search_code_rag",\n                            arguments={"query": "same"},\n                            raw_arguments='{\"query\":\"same\"}',\n                        ),\n                    )\n                )\n            assert request.tools == ()\n            assert request.tool_choice is None\n            assert request.parallel_tool_calls is False\n            assert request.media_paths == ()\n            assert request.messages[-1]["role"] == "system"\n            assert "Tool use has converged" in request.messages[-1]["content"]\n            return GenerationResponse(content="final answer from converged evidence")\n\n    adapter = LoopAdapter()\n    runtime = _ToolRuntime()\n    monkeypatch.setattr(\n        ModelRouter,\n        "_new_text_adapter",\n        staticmethod(lambda config, *, role: adapter),\n    )\n    router = ModelRouter(\n        profile="test",\n        registry=_Registry(),\n        agent_tool_runtime_factory=lambda **_: runtime,\n    )\n    assert router.generate_text(\n        "coder", [{"role": "user", "content": "research"}]\n    ) == "final answer from converged evidence"\n    assert len(runtime.calls) == 2\n    assert len(adapter.requests) == 3\n'''
    if old_test not in test_source:
        raise SystemExit("fixed-point regression test anchor not found")
    test_source = test_source.replace(old_test, new_test, 1)
    TEST.write_text(test_source, encoding="utf-8")


if __name__ == "__main__":
    main()
