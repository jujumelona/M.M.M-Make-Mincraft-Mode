from __future__ import annotations

"""Default destination and non-persistent auth discovery for MMM remote memory.

This module never grants consent and never performs a remote write. It only fills
unset destination variables and, after explicit remote-store consent, may copy a
GitHub token from Colab Secrets into the current process environment.
"""

import os

DEFAULT_REMOTE_STORE_BACKEND = "github"
DEFAULT_REMOTE_STORE_REPO = "jujumelona/mmm-data"
DEFAULT_REMOTE_STORE_BRANCH = "main"
_CONSENT_ENV = "MMM_REMOTE_TRAJECTORY_STORE_CONSENT"
_TRUE = {"1", "true", "yes", "on"}


def _consented() -> bool:
    return os.environ.get(_CONSENT_ENV, "0").strip().casefold() in _TRUE


def _load_colab_github_token_if_available() -> None:
    """Use an existing environment token or an opted-in Colab Secret, never persist it."""

    if not _consented():
        return
    if os.environ.get("MMM_TRAJECTORY_GITHUB_TOKEN", "").strip() or os.environ.get(
        "GITHUB_TOKEN", ""
    ).strip():
        return
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except Exception:
        return

    for secret_name in ("MMM_TRAJECTORY_GITHUB_TOKEN", "GITHUB_TOKEN"):
        try:
            token = str(userdata.get(secret_name) or "").strip()
        except Exception:
            continue
        if token:
            os.environ["MMM_TRAJECTORY_GITHUB_TOKEN"] = token
            return


def apply_remote_store_defaults() -> None:
    """Fill unset target variables; explicit operator values always win."""

    os.environ.setdefault("MMM_TRAJECTORY_STORE_BACKEND", DEFAULT_REMOTE_STORE_BACKEND)
    os.environ.setdefault("MMM_TRAJECTORY_STORE_REPO", DEFAULT_REMOTE_STORE_REPO)
    os.environ.setdefault("MMM_TRAJECTORY_STORE_BRANCH", DEFAULT_REMOTE_STORE_BRANCH)
    _load_colab_github_token_if_available()


__all__ = [
    "DEFAULT_REMOTE_STORE_BACKEND",
    "DEFAULT_REMOTE_STORE_BRANCH",
    "DEFAULT_REMOTE_STORE_REPO",
    "apply_remote_store_defaults",
]
