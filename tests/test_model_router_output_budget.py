from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from minecraft_mod_ai.model_router import ModelRouter


class _Adapter:
    def __init__(self) -> None:
        self.request = None

    def generate(self, request):
        self.request = request
        return "ok"


class _Router(ModelRouter):
    def __init__(self) -> None:
        self._agent_require_fresh_evidence = False
        self.adapter = _Adapter()

    def _generation_adapter(self, role: str):
        del role
        return SimpleNamespace(adapter="llama_cpp"), self.adapter

    def _tools_enabled(self, **kwargs):
        del kwargs
        return False

    @contextmanager
    def _generation_scope(self, config):
        del config
        yield


def test_generate_text_propagates_output_token_ceiling_into_request_metadata() -> None:
    router = _Router()

    assert router.generate_text(
        "coder",
        [{"role": "user", "content": "write one file"}],
        enable_tools=False,
        output_token_ceiling=4096,
    ) == "ok"

    assert router.adapter.request is not None
    assert router.adapter.request.metadata["mmm_output_token_ceiling"] == 4096
