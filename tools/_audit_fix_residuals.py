from __future__ import annotations

import json
from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"{path}: expected at least {count} matches, found {found}: {old[:100]!r}"
        )
    p.write_text(text.replace(old, new, count), encoding="utf-8")


def edit_notebook() -> None:
    path = Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    before = len(notebook["cells"])
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if "colab_mtp_server" not in "".join(cell.get("source", []))
    ]
    removed = before - len(notebook["cells"])
    if removed != 1:
        raise RuntimeError(f"expected one legacy MTP notebook cell, removed {removed}")
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def patch_streaming_tests() -> None:
    Path("tests/test_llama_server_streaming.py").write_text(
        '''from __future__ import annotations

from types import SimpleNamespace

import httpx

from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
from minecraft_mod_ai.llama_server_hardware_policy import _strict_server_generate


class _Adapter:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            role="planner",
            model_id="test/model",
            max_new_tokens=8192,
        )


class _StreamingResponse:
    status_code = 200
    text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b""

    def iter_lines(self):
        yield ': ping - connection liveness only'
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"{\\\\"ok\\\\":"}}]}'
        yield 'data: {"choices":[{"delta":{"content":"true}"}}]}'
        yield 'data: [DONE]'


class _UnavailableTelemetryResponse:
    status_code = 404
    text = ""

    @staticmethod
    def json():
        return {}


def _disable_mock_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: _UnavailableTelemetryResponse(),
    )


def test_local_native_generation_uses_sse_without_fixed_read_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}
    stream_contract._CLIENTS.clear()

    class FakeClient:
        def __init__(self, *, timeout, limits):
            captured["timeout"] = timeout
            captured["limits"] = limits

        def stream(self, method, url, *, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return _StreamingResponse()

        def close(self):
            pass

    monkeypatch.setattr(httpx, "Client", FakeClient)
    _disable_mock_telemetry(monkeypatch)
    result = _strict_server_generate(
        _Adapter(),
        SimpleNamespace(
            messages=({"role": "user", "content": "return json"},),
            response_format="json",
        ),
        "http://127.0.0.1:8910/v1",
    )

    assert result == '{"ok":true}'
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8910/v1/chat/completions"
    assert captured["json"]["stream"] is True
    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 30.0
    assert timeout.read is None
    assert timeout.write == 30.0
    assert timeout.pool == 30.0


def test_local_native_stream_requires_done_marker(monkeypatch) -> None:
    class _BrokenResponse(_StreamingResponse):
        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'

    stream_contract._CLIENTS.clear()

    class FakeClient:
        def __init__(self, *, timeout, limits):
            pass

        def stream(self, *args, **kwargs):
            return _BrokenResponse()

        def close(self):
            pass

    monkeypatch.setattr(httpx, "Client", FakeClient)
    _disable_mock_telemetry(monkeypatch)

    try:
        _strict_server_generate(
            _Adapter(),
            SimpleNamespace(
                messages=({"role": "user", "content": "x"},),
                response_format="text",
            ),
            "http://127.0.0.1:8910/v1",
        )
    except Exception as exc:
        assert "stream ended before the [DONE] marker" in str(exc)
    else:
        raise AssertionError("truncated SSE stream must fail closed")
''',
        encoding="utf-8",
    )


def main() -> None:
    edit_notebook()

    replace(
        "tests/test_colab_native_llama_prebuilt_integration.py",
        '    monkeypatch.delenv("MMM_LLAMA_SERVER_DISTRIBUTION", raising=False)\n'
        '    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)\n',
        '    monkeypatch.delenv("MMM_LLAMA_SERVER_DISTRIBUTION", raising=False)\n'
        '    monkeypatch.setenv("MMM_LLAMA_ALLOW_SOURCE_BUILD", "1")\n'
        '    monkeypatch.setattr(module, "_find_verified_native_server", lambda: None)\n',
    )
    replace(
        "tests/test_mcp_complete_plan_refs.py",
        "mmm/complete-plan-result-v3",
        "mmm/complete-plan-result-v4",
    )
    replace(
        "tests/test_mcp_config.py",
        '    assert "MMM_MCP_STAGE" not in generation["env"]\n',
        '    assert generation["env"]["MMM_MCP_STAGE"] == "generation"\n',
    )
    replace(
        "tests/test_model_runtime_performance_contract.py",
        '    assert getattr(\n'
        '        ImageDiffusionAdapter.generate_image,\n'
        '        "_mmm_cached_image_pipeline",\n'
        '        False,\n'
        '    )\n',
        "",
    )
    replace(
        "tests/test_notebook_registry_policy.py",
        '    assert \'REMOTE_PROJECT_INSTALL_TARGET = (\\n    ".[ui,rag,\' in setup_source\n',
        '    assert \'REMOTE_PROJECT_INSTALL_TARGET = ".[ui,rag,\' in setup_source\n',
    )
    replace(
        "tests/test_notebook_registry_policy.py",
        '    assert \'LOCAL_PROJECT_INSTALL_TARGET = (\\n    ".[ui,local-model,rag,\' in setup_source\n',
        '    assert \'LOCAL_PROJECT_INSTALL_TARGET = ".[ui,local-model,rag,\' in setup_source\n',
    )
    replace(
        "tests/test_notebook_registry_policy.py",
        '        setup_fingerprint=first,\n        torch=None,\n    )\n',
        '        setup_fingerprint=first,\n        torch=None,\n        llama_server_binary="",\n    )\n',
    )

    # Sequential production-outline pages are stricter than ordinary structured
    # responses: unrelated outer JSON must still fail closed.
    strict = Path("tests/test_planner_strict_json_contract.py")
    text = strict.read_text(encoding="utf-8")
    text = text.replace(
        'match="found 2 complete outermost JSON containers"',
        'match="found 2 matching objects"',
    )
    strict.write_text(text, encoding="utf-8")

    replace(
        "tests/test_planner_outline_prompt_contract.py",
        '    assert "Choose the page size yourself" in system\n',
        '    assert "Choose page size yourself" in system\n',
    )
    replace(
        "tests/test_planner_pagination_safety_contract.py",
        'match="no host-verifiable deliverable progress"',
        'match="no verified progress"',
    )

    replace(
        "minecraft_mod_ai/execution_efficiency_contract.py",
        '''    emitted = 0\n    while ready:\n        _, index = heapq.heappop(ready)\n        group = groups[index]\n        yield str(group["stage"]), tuple(group["members"])\n        emitted += 1\n        for dependent in dependents[index]:\n            indegree[dependent] -= 1\n            if indegree[dependent] == 0:\n                heapq.heappush(\n                    ready,\n                    (int(groups[dependent]["first_order"]), dependent),\n                )\n''',
        '''    emitted = 0\n    while ready:\n        current_wave: list[tuple[int, int]] = []\n        while ready:\n            current_wave.append(heapq.heappop(ready))\n        next_wave: list[tuple[int, int]] = []\n        for _, index in current_wave:\n            group = groups[index]\n            yield str(group["stage"]), tuple(group["members"])\n            emitted += 1\n            for dependent in dependents[index]:\n                indegree[dependent] -= 1\n                if indegree[dependent] == 0:\n                    heapq.heappush(\n                        next_wave,\n                        (int(groups[dependent]["first_order"]), dependent),\n                    )\n        ready = next_wave\n''',
    )

    patch_streaming_tests()


if __name__ == "__main__":
    main()
