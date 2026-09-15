from __future__ import annotations

"""Optional versioned cross-run trajectory persistence outside the MMM source repo.

Writes are fail-closed behind explicit Colab consent and verifier qualification.
Production never depends on remote persistence: eligible records are queued locally
first and a failed sync leaves the outbox intact.
"""

import base64
import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .remote_skill_store_consent import (
    remote_write_allowed,
    require_remote_write_consent,
    sanitize_remote_payload,
)
from .trajectory_memory import memory_path, remote_cache_path
from .trajectory_record_integrity import record_remote_eligible
from .trajectory_verification import REMOTE_FORMAT_VERSION, TRAJECTORY_SCHEMA_VERSION

_BACKEND_ENV = "MMM_TRAJECTORY_STORE_BACKEND"
_REPO_ENV = "MMM_TRAJECTORY_STORE_REPO"
_BRANCH_ENV = "MMM_TRAJECTORY_STORE_BRANCH"
_MAX_REMOTE_ROWS = 5000
_REMOTE_ID_KEY = "remote_record_id"
_REMOTE_FORMAT_KEY = "remote_format_version"
_TASK_CLASSES = (
    "repair",
    "generation",
    "build",
    "runtime",
    "quality",
    "research",
    "planning",
    "release",
    "general",
)


def _outbox(base: str | Path) -> Path:
    return memory_path(base).parent / "remote-outbox.jsonl"


def _backend() -> str:
    value = os.environ.get(_BACKEND_ENV, "none").strip().casefold()
    return value if value in {"none", "github", "huggingface"} else "none"


def _raw_repo() -> str:
    return os.environ.get(_REPO_ENV, "").strip()


def _normalize_repo(value: str, backend: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    if "://" not in raw:
        return raw.removesuffix(".git").strip("/")
    parsed = urllib.parse.urlparse(raw)
    segments = [segment for segment in parsed.path.split("/") if segment]
    host = parsed.netloc.casefold()
    if backend == "github" and host in {"github.com", "www.github.com"} and len(segments) >= 2:
        return f"{segments[0]}/{segments[1].removesuffix('.git')}"
    if backend == "huggingface" and host in {"huggingface.co", "www.huggingface.co"}:
        if segments and segments[0] in {"datasets", "models", "spaces"}:
            segments = segments[1:]
        if len(segments) >= 2:
            return f"{segments[0]}/{segments[1]}"
    return raw.removesuffix(".git").strip("/")


def _repo() -> str:
    return _normalize_repo(_raw_repo(), _backend())


def _branch() -> str:
    return os.environ.get(_BRANCH_ENV, "main").strip() or "main"


def _hf_repo_type() -> str:
    explicit = os.environ.get("MMM_TRAJECTORY_HF_REPO_TYPE", "").strip().casefold()
    if explicit in {"model", "dataset", "space"}:
        return explicit
    raw = _raw_repo().casefold()
    if "/spaces/" in raw:
        return "space"
    if "/models/" in raw:
        return "model"
    return "dataset"


def _remote_path(task_class: str) -> str:
    return f"memory/{REMOTE_FORMAT_VERSION}/{task_class}.jsonl"


def _manifest_path() -> str:
    return f"memory/{REMOTE_FORMAT_VERSION}/manifest.json"


def _legacy_remote_path(task_class: str) -> str:
    return f"memory/{task_class}.jsonl"


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": "mmm/trajectory-store-manifest-v1",
        "format_version": REMOTE_FORMAT_VERSION,
        "trajectory_schema": TRAJECTORY_SCHEMA_VERSION,
        "storage": "jsonl-sharded-by-task-class",
        "shard_pattern": f"memory/{REMOTE_FORMAT_VERSION}/{{task_class}}.jsonl",
        "privacy_policy": "sanitized-structural-verifier-evidence-only",
        "success_policy": "remote-success-requires-L3-or-higher",
        "failure_policy": "remote-failure-requires-objective-verifier-failure",
        "task_classes": list(_TASK_CLASSES),
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _stamp_remote_record(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(sanitize_remote_payload(dict(row)))
    value.pop(_REMOTE_ID_KEY, None)
    value[_REMOTE_FORMAT_KEY] = REMOTE_FORMAT_VERSION
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    value[_REMOTE_ID_KEY] = "sha256:" + digest
    return value


def _remote_hash_valid(row: Mapping[str, Any]) -> bool:
    identity = str(row.get(_REMOTE_ID_KEY, ""))
    if not identity.startswith("sha256:") or len(identity) != 71:
        return False
    candidate = dict(row)
    candidate.pop(_REMOTE_ID_KEY, None)
    expected = "sha256:" + hashlib.sha256(_canonical_bytes(candidate)).hexdigest()
    return identity == expected and row.get(_REMOTE_FORMAT_KEY) == REMOTE_FORMAT_VERSION


def _remote_record_valid(row: Mapping[str, Any]) -> bool:
    return _remote_hash_valid(row) and record_remote_eligible(row)


def _github_token() -> str:
    return os.environ.get("MMM_TRAJECTORY_GITHUB_TOKEN", "").strip() or os.environ.get("GITHUB_TOKEN", "").strip()


def _hf_token() -> str:
    return os.environ.get("MMM_TRAJECTORY_HF_TOKEN", "").strip() or os.environ.get("HF_TOKEN", "").strip()


def _backend_auth_available() -> bool:
    """Return whether the selected backend can actually make authenticated calls."""

    backend = _backend()
    if backend == "github":
        return bool(_github_token())
    if backend == "huggingface":
        return bool(_hf_token())
    return False


def remote_configured() -> bool:
    """Return True only when consent, destination and backend credentials are ready."""

    return (
        remote_write_allowed()
        and _backend() != "none"
        and bool(_repo())
        and _backend_auth_available()
    )


def _iter_outbox_lines_reverse(path: Path, *, block_size: int = 64 * 1024):
    """Yield outbox lines newest-first without scanning the whole file."""

    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        carry = b""
        while position > 0:
            size = min(block_size, position)
            position -= size
            handle.seek(position)
            parts = (handle.read(size) + carry).split(b"\n")
            carry = parts[0]
            for raw in reversed(parts[1:]):
                if raw:
                    yield raw.decode("utf-8")
        if carry:
            yield carry.decode("utf-8")


def queue_remote_record(base: str | Path, row: Mapping[str, Any]) -> bool:
    """Queue one sanitized, verifier-qualified record only after explicit opt-in."""

    if not remote_write_allowed() or not record_remote_eligible(row):
        return False
    stamped = _stamp_remote_record(row)
    if not _remote_record_valid(stamped):
        return False
    path = _outbox(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = str(stamped.get(_REMOTE_ID_KEY, ""))
    recent: set[str] = set()
    if path.is_file() and not path.is_symlink():
        try:
            recent_rows = 0
            for raw in _iter_outbox_lines_reverse(path):
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, Mapping):
                    recent.add(str(value.get(_REMOTE_ID_KEY, "")))
                    recent_rows += 1
                    if recent_rows >= 512:
                        break
        except OSError:
            return False
    if identity in recent:
        return False
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(stamped, ensure_ascii=False, sort_keys=True) + "\n")
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
            if isinstance(value, dict) and _remote_record_valid(value):
                rows.append(value)
    return list(rows)


def _merge_rows(existing: Sequence[Mapping[str, Any]], pending: Sequence[Mapping[str, Any]]) -> bytes:
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in [*existing, *pending]:
        if not _remote_record_valid(row):
            continue
        identity = str(row.get(_REMOTE_ID_KEY, ""))
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
    token = _github_token()
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


def _github_get_bytes(path: str) -> tuple[bytes, str]:
    encoded_path = urllib.parse.quote(path, safe="/")
    ref = urllib.parse.quote(_branch(), safe="")
    value = _github_request("GET", f"/repos/{_repo()}/contents/{encoded_path}?ref={ref}")
    if value is None:
        return b"", ""
    return (
        base64.b64decode(str(value.get("content", "")).encode("ascii")),
        str(value.get("sha", "")),
    )


def _github_put_bytes(path: str, content: bytes, *, message: str) -> None:
    require_remote_write_consent()
    current, sha = _github_get_bytes(path)
    if current == content and sha:
        return
    encoded_path = urllib.parse.quote(path, safe="/")
    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": _branch(),
    }
    if sha:
        body["sha"] = sha
    _github_request("PUT", f"/repos/{_repo()}/contents/{encoded_path}", body=body)


def _github_read_path(path: str) -> tuple[list[dict[str, Any]], str]:
    content, sha = _github_get_bytes(path)
    if not content:
        return [], sha
    rows: list[dict[str, Any]] = []
    for raw in content.decode("utf-8", errors="replace").splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and _remote_record_valid(item):
            rows.append(item)
    return rows[-_MAX_REMOTE_ROWS:], sha


def _github_read(task_class: str) -> tuple[list[dict[str, Any]], str]:
    rows, sha = _github_read_path(_remote_path(task_class))
    if rows or sha:
        return rows, sha
    # Pre-v3 files are never trusted as procedural memory because they lack a
    # versioned verifier chain and remote content hash.
    return [], ""


def _github_write(task_class: str, pending: Sequence[Mapping[str, Any]]) -> None:
    existing, _sha = _github_read(task_class)
    _github_put_bytes(
        _remote_path(task_class),
        _merge_rows(existing, pending),
        message=f"Update {REMOTE_FORMAT_VERSION} verified {task_class} trajectories",
    )


def _github_write_manifest() -> None:
    _github_put_bytes(
        _manifest_path(),
        json.dumps(_manifest(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        message=f"Update MMM trajectory store manifest {REMOTE_FORMAT_VERSION}",
    )


def _hf_read_path(path_in_repo: str) -> list[dict[str, Any]]:
    token = _hf_token()
    if not token:
        raise RuntimeError("Hugging Face trajectory store requires HF_TOKEN or MMM_TRAJECTORY_HF_TOKEN.")
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            _repo(),
            filename=path_in_repo,
            repo_type=_hf_repo_type(),
            token=token,
        )
    except Exception as exc:
        text = str(exc).casefold()
        if "404" in text or "entry not found" in text:
            return []
        raise
    return _read_jsonl(Path(path))


def _hf_read(task_class: str) -> list[dict[str, Any]]:
    return _hf_read_path(_remote_path(task_class))


def _hf_upload_bytes(path_in_repo: str, content: bytes, *, message: str) -> None:
    require_remote_write_consent()
    token = _hf_token()
    if not token:
        raise RuntimeError("Hugging Face trajectory store requires HF_TOKEN or MMM_TRAJECTORY_HF_TOKEN.")
    from huggingface_hub import HfApi

    with tempfile.NamedTemporaryFile("wb", suffix=Path(path_in_repo).suffix or ".bin", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        HfApi(token=token).upload_file(
            path_or_fileobj=str(temporary),
            path_in_repo=path_in_repo,
            repo_id=_repo(),
            repo_type=_hf_repo_type(),
            commit_message=message,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _hf_write(task_class: str, pending: Sequence[Mapping[str, Any]]) -> None:
    _hf_upload_bytes(
        _remote_path(task_class),
        _merge_rows(_hf_read(task_class), pending),
        message=f"Update {REMOTE_FORMAT_VERSION} verified {task_class} trajectories",
    )


def _hf_write_manifest() -> None:
    _hf_upload_bytes(
        _manifest_path(),
        json.dumps(_manifest(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        message=f"Update MMM trajectory store manifest {REMOTE_FORMAT_VERSION}",
    )


def hydrate_remote_cache(base: str | Path, task_class: str) -> bool:
    """Refresh one verified task-class shard; failures never break production."""

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
    """Flush qualified records at a safe work-unit boundary, keeping failures local."""

    require_remote_write_consent()
    backend = _backend()
    repo = _repo()
    if backend == "none" or not repo:
        return {"status": "NOT_CONFIGURED", "flushed": 0}
    path = _outbox(base)
    rows = _read_jsonl(path, max_rows=10000)
    if not rows:
        path.unlink(missing_ok=True)
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
        if backend == "github":
            _github_write_manifest()
        else:
            _hf_write_manifest()
    except Exception as exc:
        return {"status": "DEFERRED", "flushed": 0, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    path.unlink(missing_ok=True)
    return {
        "status": "SYNCED",
        "flushed": len(rows),
        "backend": backend,
        "repo": repo,
        "format_version": REMOTE_FORMAT_VERSION,
        "manifest": _manifest_path(),
    }


__all__ = [
    "flush_remote_outbox",
    "hydrate_remote_cache",
    "queue_remote_record",
    "remote_configured",
]
