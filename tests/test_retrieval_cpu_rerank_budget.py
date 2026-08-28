from __future__ import annotations

from minecraft_mod_ai import model_router


class _RoleConfig:
    adapter = "reranker"

    def __init__(self, device: str) -> None:
        self.extra = {"device": device}


class _Registry:
    def __init__(self, device: str) -> None:
        self.device = device

    def role(self, profile: str, role: str):
        del profile, role
        return _RoleConfig(self.device)


class _RouterLike:
    def __init__(self, device: str) -> None:
        self.profile = "test"
        self.registry = _Registry(device)


def _install_sentinel(monkeypatch):
    constructions: list[str] = []
    calls: list[tuple[str, tuple[str, ...]]] = []

    class SentinelReranker:
        def __init__(self, config) -> None:
            constructions.append(str(config.extra.get("device")))

        def score(self, query: str, documents, *, instruction: str):
            del instruction
            values = tuple(str(item) for item in documents)
            calls.append((query, values))
            return [float(index + 1) for index in range(len(values))]

    monkeypatch.setattr(model_router, "RerankerAdapter", SentinelReranker)
    return constructions, calls


def test_disabled_cpu_reranker_is_skipped_without_backend_attempt(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    constructions, calls = _install_sentinel(monkeypatch)

    scores = model_router.ModelRouter.rerank(
        _RouterLike("cpu"),
        "alien planet",
        ["one", "two"],
    )

    assert scores == []
    assert constructions == []
    assert calls == []


def test_explicit_cpu_dense_opt_in_delegates_to_reranker(monkeypatch) -> None:
    monkeypatch.setenv("MMM_RAG_ENABLE_CPU_DENSE", "1")
    constructions, calls = _install_sentinel(monkeypatch)

    scores = model_router.ModelRouter.rerank(
        _RouterLike("cpu"),
        "alien planet",
        ["one", "two"],
    )

    assert scores == [1.0, 2.0]
    assert constructions == ["cpu"]
    assert calls == [("alien planet", ("one", "two"))]


def test_non_cpu_reranker_is_not_suppressed(monkeypatch) -> None:
    monkeypatch.delenv("MMM_RAG_ENABLE_CPU_DENSE", raising=False)
    constructions, calls = _install_sentinel(monkeypatch)

    scores = model_router.ModelRouter.rerank(
        _RouterLike("cuda"),
        "alien planet",
        ["one"],
    )

    assert scores == [1.0]
    assert constructions == ["cuda"]
    assert calls == [("alien planet", ("one",))]
