from __future__ import annotations

"""Default destination for optional MMM remote procedural-memory persistence.

This module only supplies a destination. It never grants consent and never performs
network I/O. The independent consent gate remains authoritative for every read/write
path that can reach the remote trajectory store.
"""

import os

DEFAULT_REMOTE_STORE_BACKEND = "github"
DEFAULT_REMOTE_STORE_REPO = "jujumelona/mmm-data"
DEFAULT_REMOTE_STORE_BRANCH = "main"


def apply_remote_store_defaults() -> None:
    """Fill unset destination variables without overriding operator choices."""

    os.environ.setdefault("MMM_TRAJECTORY_STORE_BACKEND", DEFAULT_REMOTE_STORE_BACKEND)
    os.environ.setdefault("MMM_TRAJECTORY_STORE_REPO", DEFAULT_REMOTE_STORE_REPO)
    os.environ.setdefault("MMM_TRAJECTORY_STORE_BRANCH", DEFAULT_REMOTE_STORE_BRANCH)


__all__ = [
    "DEFAULT_REMOTE_STORE_BACKEND",
    "DEFAULT_REMOTE_STORE_BRANCH",
    "DEFAULT_REMOTE_STORE_REPO",
    "apply_remote_store_defaults",
]
