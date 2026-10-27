from __future__ import annotations

import os

from minecraft_mod_ai import remote_trajectory_store as store
from minecraft_mod_ai.remote_skill_store_consent import (
    remote_write_allowed,
    require_remote_write_consent,
    sanitize_remote_payload,
)
from minecraft_mod_ai.remote_store_defaults import (
    DEFAULT_REMOTE_STORE_BACKEND,
    DEFAULT_REMOTE_STORE_BRANCH,
    DEFAULT_REMOTE_STORE_REPO,
    apply_remote_store_defaults,
    retired_remote_repository,
)


def _clear_target(monkeypatch) -> None:
    for name in (
        "MMM_TRAJECTORY_STORE_BACKEND",
        "MMM_TRAJECTORY_STORE_REPO",
        "MMM_TRAJECTORY_STORE_BRANCH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_remote_store_defaults_are_disabled(monkeypatch) -> None:
    _clear_target(monkeypatch)
    monkeypatch.delenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", raising=False)
    apply_remote_store_defaults()
    assert os.environ["MMM_TRAJECTORY_STORE_BACKEND"] == DEFAULT_REMOTE_STORE_BACKEND == "none"
    assert os.environ["MMM_TRAJECTORY_STORE_REPO"] == DEFAULT_REMOTE_STORE_REPO == ""
    assert os.environ["MMM_TRAJECTORY_STORE_BRANCH"] == DEFAULT_REMOTE_STORE_BRANCH == "main"
    assert remote_write_allowed() is False


def test_destination_defaults_never_grant_remote_consent(monkeypatch, tmp_path) -> None:
    _clear_target(monkeypatch)
    monkeypatch.delenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", raising=False)
    apply_remote_store_defaults()

    assert remote_write_allowed() is False
    assert store.remote_configured() is False
    assert store.queue_remote_record(tmp_path, {"task_class": "repair"}) is False
    assert not (tmp_path / ".minecraft_ai" / "trajectory-memory" / "remote-outbox.jsonl").exists()


def test_consent_without_destination_does_not_enable_remote_access(monkeypatch) -> None:
    _clear_target(monkeypatch)
    monkeypatch.setenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", "1")
    monkeypatch.delenv("MMM_TRAJECTORY_GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    apply_remote_store_defaults()
    assert remote_write_allowed() is True
    assert store.remote_configured() is False
    assert os.environ["MMM_TRAJECTORY_STORE_BACKEND"] == "none"
    assert os.environ["MMM_TRAJECTORY_STORE_REPO"] == ""

def test_retired_destination_is_blocked_even_with_consent_and_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", "1")
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_BACKEND", "github")
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_REPO", "https://github.com/jujumelona/mmm-data.git")
    monkeypatch.setenv("MMM_TRAJECTORY_GITHUB_TOKEN", "test-token")
    apply_remote_store_defaults()
    assert retired_remote_repository() is True
    assert remote_write_allowed() is False
    assert store.remote_configured() is False
    assert store.queue_remote_record(tmp_path, {"task_class": "repair"}) is False
    try:
        require_remote_write_consent()
    except PermissionError:
        pass
    else:
        raise AssertionError("Retired destination was not blocked")


def test_explicit_operator_store_target_is_not_overridden(monkeypatch) -> None:
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_BACKEND", "huggingface")
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_REPO", "owner/custom-memory")
    monkeypatch.setenv("MMM_TRAJECTORY_STORE_BRANCH", "verified")
    apply_remote_store_defaults()

    assert os.environ["MMM_TRAJECTORY_STORE_BACKEND"] == "huggingface"
    assert os.environ["MMM_TRAJECTORY_STORE_REPO"] == "owner/custom-memory"
    assert os.environ["MMM_TRAJECTORY_STORE_BRANCH"] == "verified"


def test_remote_sanitizer_still_removes_source_prompts_and_credentials() -> None:
    sanitized = sanitize_remote_payload(
        {
            "task_class": "repair",
            "verification_level": 3,
            "prompt": "private request",
            "source_code": "private source",
            "github_token": "secret",
            "nested": {"password": "secret", "status": "PASS"},
        }
    )

    assert sanitized == {
        "task_class": "repair",
        "verification_level": 3,
        "nested": {"status": "PASS"},
    }
