from __future__ import annotations

"""Optional cross-run trajectory persistence outside the MMM source repository.

Writes are fail-closed behind explicit Colab consent. Production never depends on
remote persistence: records are first queued locally and a failed sync leaves the
outbox intact.
"""

import base64
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from .remote_skill_store_consent import (
    remote_write_allowed,
    require_remote_write_consent,
    sanitize_remote_payload,
)
from .trajectory_memory import remote_cache_path

_BACKEND_ENV = "MMM_TRAJECTORY_STORE_BACKEND"
_REPO_ENV = "MMM_TRAJECTORY_STORE_REPO"
_BRANCH_ENV = "MMM_TRAJECTORY_STORE_BRANCH"
_MAX_REMOTE_ROWS = 5000


def _outbox(base: str | Path) -> Path:
    return Path(base).expanduser().resolve() / ".minecraft_ai" / "trajectory-memory" / "remote-outbox.jsonl"


def _backend() -> str:
    value = os.environ.get(_BACKEND_ENV, "none").strip().casefold()
    return value if value in {"none", "github", "huggingface"} else "none"


def _repo() -> str:
    return os.environ.get(_REPO_ENV, "").strip()


def remote_configured() -> bool:
    return remote_write_allowed() and _backend() != "none" and bool(_repo())


def queue_remote_record(base: str | Path, row: Mapping[str, Any]) -> bool:
    """Queue one sanitized record only when the user explicitly opted in."""

    if not remote_write_allowed():
        return False
    sanitized = sanitize_remote_payload(dict(row))
    path = _outbox(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = str(sanitized.get("trajectory_id", ""))
    if not identity:
        return False
    recent: deque[str] = deque(maxlen=512)
    if path.is_file() and not path.is_symlink():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, Mapping):
                        recent.append(str(value.get("trajectory_id", "")))
        except OSError:
            return False
    if identity in recent:
        return False
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sanitized, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def _read_jsonl(path: Path, *, max_rows: int = _MAX_REMOTE_ROWS) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return list(rows)


def _merge_rows(existing: Sequence[Mapping[str, Any]], pending: Sequence[Mapping[str, Any]]) -> bytes:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in [*existing, *pending]:
        identity = str(row.get("trajectory_id", ""))
        if not identity:
            continue
        if identity not in by_id:
            order.append(identity)
        by_id[identity] = dict(row)
    order = order[-_MAX_REMOTE_ROWS:]
    text = "".join(
        json.dumps(by_id[identity], ensure_ascii=False, sort_keys=True) + "\n"
        for identity in order
    )
    return text.encode("utf-8")


def _github_request(method: str, path: str, *, body: Mapping[str, Any] | None = None) -> Any:
    token = os.environ.get("MMM_TRAJECTORY_GITHUB_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GitHub trajectory store requires GITHUB_TOKEN or MMM_TRAJECTORY_GITHUB_TOKEN.")
    url = "https://api.github.com" + path
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "mmm-trajectory-store",
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and method == "GET":
            return None
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"GitHub trajectory store HTTP {exc.code}: {detail}") from exc
    return json.loads(raw.decode("utf-8")) if raw else {}


def _github_read(task_class: str) -> tuple[list[dict[str, Any]], str]:
    repo = _repo()
    path = f"memory/{task_class}.jsonl"
    encoded_path = urllib.parse.quote(path, safe="/")
    value = _github_request("GET", f"/repos/{repo}/contents/{encoded_path}")
    if value is None:
        return [], ""
    content = base64.b64decode(str(value.get("content", "")).encode("ascii"))
    rows: list[dict[str, Any]] = []
    for raw in content.decode("utf-8", errors="replace").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows[-_MAX_REMOTE_ROWS:], str(value.get("sha", ""))


def _github_write(task_class: str, pending: Sequence[Mapping[str, Any]]) -> None:
    require_remote_write_consent()
    existing, sha = _github_read(task_class)
    content = _merge_rows(existing, pending)
    repo = _repo()
    path = f"memory/{task_class}.jsonl"
    encoded_path = urllib.parse.quote(path, safe="/")
    body: dict[str, Any] = {
        "message": f"Update verified {task_class} trajectories",
        "content": base64.b64encode(content).decode("ascii"),
        "branch": os.environ.get(_BRANCH_ENV, "main").strip() or "main",
    }
    if sha:
        body["sha"] = sha
    _github_request("PUT", f"/repos/{repo}/contents/{encoded_path}", body=body)


def _hf_token() -> str:
    return os.environ.get("MMM_TRAJECTORY_HF_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()


def _hf_read(task_class: str) -> list[dict[str, Any]]:
    token = _hf_token()
    if not token:
        raise RuntimeError("Hugging Face trajectory store requires HF_TOKEN or MMM_TRAJECTORY_HF_TOKEN.")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            _repo(),
            filename=f"memory/{task_class}.jsonl",
            repo_type=os.environ.get("MMM_TRAJECTORY_HF_REPO_TYPE", "dataset"),
            token=token,
        )
    except Exception as exc:
        text = str(exc).casefold()
        if "404" in text or "entry not found" in text:
            return []
        raise
    return _read_jsonl(Path(path))


def _hf_write(task_class: str, pending: Sequence[Mapping[str, Any]]) -> None:
    require_remote_write_consent()
    token = _hf_token()
    if not token:
        raise RuntimeError("Hugging Face trajectory store requires HF_TOKEN or MMM_TRAJECTORY_HF_TOKEN.")
    from huggingface_hub import HfApi

    content = _merge_rows(_hf_read(task_class), pending)
    with tempfile.NamedTemporaryFile("wb", suffix=".jsonl", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=str(temporary),
            path_in_repo=f"memory/{task_class}.jsonl",
            repo_id=_repo(),
            repo_type=os.environ.get("MMM_TRAJECTORY_HF_REPO_TYPE", "dataset"),
            commit_message=f"Update verified {task_class} trajectories",
        )
    finally:
        temporary.unlink(missing_ok=True)


def hydrate_remote_cache(base: str | Path, task_class: str) -> bool:
    """Refresh one task-class shard; failures never break production."""

    if not remote_configured():
        return False
    try:
        rows = _github_read(task_class)[0] if _backend() == "github" else _hf_read(task_class)
        target = remote_cache_path(base, task_class)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_merge_rows([], rows))
        return True
    except Exception as exc:
        print("trajectory remote cache skipped:", f"{type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return False


def flush_remote_outbox(base: str | Path) -> dict[str, Any]:
    """Flush queued records at a safe work-unit boundary, keeping failures local."""

    require_remote_write_consent()
    backend = _backend()
    repo = _repo()
    if backend == "none" or not repo:
        return {"status": "NOT_CONFIGURED", "flushed": 0}
    path = _outbox(base)
    rows = _read_jsonl(path, max_rows=10000)
    if not rows:
        return {"status": "EMPTY", "flushed": 0}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task_class", "general"))].append(row)
    try:
        for task_class, values in sorted(grouped.items()):
            if backend == "github":
                _github_write(task_class, values)
            else:
                _hf_write(task_class, values)
    except Exception as exc:
        return {"status": "DEFERRED", "flushed": 0, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    path.unlink(missing_ok=True)
    return {"status": "SYNCED", "flushed": len(rows), "backend": backend, "repo": repo}


__all__ = [
    "flush_remote_outbox",
    "hydrate_remote_cache",
    "queue_remote_record",
    "remote_configured",
]
