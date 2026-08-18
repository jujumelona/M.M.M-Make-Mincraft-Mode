from __future__ import annotations

from minecraft_mod_ai.remote_trajectory_store import remote_configured


def _enable_remote(monkeypatch, *, backend: str) -> None:
    monkeypatch.setenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", "1")
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_BACKEND", backend)
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_REPO", "example/trajectory-memory")
    monkeypatch.delenv("MMM_TRAJECTORY_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("MMM_TRAJECTORY_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_github_remote_cache_is_not_configured_without_token(monkeypatch) -> None:
    _enable_remote(monkeypatch, backend="github")
    assert remote_configured() is False

    monkeypatch.setenv("MMM_TRAJECTORY_GITHUB_TOKEN", "test-token")
    assert remote_configured() is True


def test_huggingface_remote_cache_is_not_configured_without_token(monkeypatch) -> None:
    _enable_remote(monkeypatch, backend="huggingface")
    assert remote_configured() is False

    monkeypatch.setenv("MMM_TRAJECTORY_HF_TOKEN", "test-token")
    assert remote_configured() is True
