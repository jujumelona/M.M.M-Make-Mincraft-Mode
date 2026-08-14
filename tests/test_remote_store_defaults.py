from __future__ import annotations

import os

from minecraft_mod_ai import remote_trajectory_store as store
from minecraft_mod_ai.remote_skill_store_consent import (
    remote_write_allowed,
    sanitize_remote_payload,
)
from minecraft_mod_ai.remote_store_defaults import (
    DEFAULT_REMOTE_STORE_BACKEND,
    DEFAULT_REMOTE_STORE_BRANCH,
    DEFAULT_REMOTE_STORE_REPO,
    apply_remote_store_defaults,
)


def _clear_target(monkeypatch) -> None:
    for name in (
        "MMM_TRAJECTORY_STORE_BACKEND",
        "MMM_TRAJECTORY_STORE_REPO",
        "MMM_TRAJECTORY_STORE_BRANCH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_mmm_data_is_the_default_remote_store_target(monkeypatch) -> None:
    _clear_target(monkeypatch)
    monkeypatch.delenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", raising=False)
    apply_remote_store_defaults()

    assert os.environ["MMM_TRAJECTORY_STORE_BACKEND"] == DEFAULT_REMOTE_STORE_BACKEND == "github"
    assert os.environ["MMM_TRAJECTORY_STORE_REPO"] == DEFAULT_REMOTE_STORE_REPO == "jujumelona/mmm-data"
    assert os.environ["MMM_TRAJECTORY_STORE_BRANCH"] == DEFAULT_REMOTE_STORE_BRANCH == "main"


def test_destination_defaults_never_grant_remote_consent(monkeypatch, tmp_path) -> None:
    _clear_target(monkeypatch)
    monkeypatch.delenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", raising=False)
    apply_remote_store_defaults()

    assert remote_write_allowed() is False
    assert store.remote_configured() is False
    assert store.queue_remote_record(tmp_path, {"task_class": "repair"}) is False
    assert not (tmp_path / ".minecraft_ai" / "trajectory-memory" / "remote-outbox.jsonl").exists()


def test_explicit_consent_activates_default_mmm_data_destination(monkeypatch) -> None:
    _clear_target(monkeypatch)
    monkeypatch.setenv("MMM_REMOTE_TRAJECTORY_STORE_CONSENT", "1")
    apply_remote_store_defaults()

    assert remote_write_allowed() is True
    assert store.remote_configured() is True
    assert os.environ["MMM_TRAJECTORY_STORE_BACKEND"] == "github"
    assert os.environ["MMM_TRAJECTORY_STORE_REPO"] == "jujumelona/mmm-data"


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
