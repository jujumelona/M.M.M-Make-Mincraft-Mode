from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.generation_verifier_runtime_contract import install


def test_install_only_owns_persistent_jdt_lifecycle() -> None:
    closed: list[str] = []

    class Runtime:
        def close(self) -> None:
            closed.append("runtime")

    class Service:
        def close(self) -> None:
            closed.append("jdt")

    verifier = SimpleNamespace()
    runtime_module = SimpleNamespace(AgentToolRuntime=Runtime)
    install(agent_tool_runtime_module=runtime_module, verifier_module=verifier)

    runtime = Runtime()
    runtime._mmm_generation_java_service = Service()
    runtime.close()

    assert closed == ["jdt", "runtime"]
    assert not hasattr(runtime, "_mmm_generation_java_service")
    assert not hasattr(verifier, "_run_gradle_fallback")


def test_install_is_idempotent() -> None:
    calls: list[str] = []

    class Runtime:
        def close(self) -> None:
            calls.append("runtime")

    runtime_module = SimpleNamespace(AgentToolRuntime=Runtime)
    verifier = SimpleNamespace()
    install(agent_tool_runtime_module=runtime_module, verifier_module=verifier)
    first_close = Runtime.close
    install(agent_tool_runtime_module=runtime_module, verifier_module=verifier)

    assert Runtime.close is first_close
    Runtime().close()
    assert calls == ["runtime"]
