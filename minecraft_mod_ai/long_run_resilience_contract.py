from __future__ import annotations

"""Late-runtime durability for long-running planner research.

The pre-design pipeline can legitimately run for many model calls. This contract keeps
that work both bounded and resumable without weakening evidence coverage:

* document-backed synthesis never re-enters RAG/MCP after every evidence page was read;
* successful no-tool research generations are checkpointed by their exact full request;
* repeated synthesis over unchanged evidence reaches a host-side fixed point without
  another model call;
* an MMM-managed llama.cpp process may be restarted after it dies, and the interrupted
  request is replayed exactly once.

No prompt/evidence text is truncated or omitted by this layer.
"""

import hashlib
import json
import os
import tempfile
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_long_run_resilience_v1"
_CACHE_SCHEMA = "mmm/research-generation-checkpoint-v1"
_CACHE_ROOT_ENV = "MMM_RESEARCH_CHECKPOINT_ROOT"
_CACHE_LOCK = threading.RLock()
_KEY_LOCKS: dict[str, threading.Lock] = {}
_SYNTHESIS_LOCK = threading.RLock()
_SYNTHESIS_RESULTS: dict[str, str] = {}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _checkpoint_root() -> Path:
    configured = (os.environ.get(_CACHE_ROOT_ENV) or "").strip()
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(tempfile.gettempdir()) / "mmm-research-checkpoints-v1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _key_lock(key: str) -> threading.Lock:
    with _CACHE_LOCK:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


def _role_signature(router: Any, role: str) -> dict[str, Any]:
    signature: dict[str, Any] = {
        "router": f"{type(router).__module__}.{type(router).__qualname__}",
        "profile": str(getattr(router, "profile", "")),
        "role": role,
    }
    registry = getattr(router, "registry", None)
    resolver = getattr(registry, "role", None)
    if not callable(resolver):
        return signature
    try:
        config = resolver(getattr(router, "profile", ""), role)
    except Exception:
        return signature
    for name in (
        "model",
        "model_id",
        "adapter",
        "backend",
        "max_context",
        "max_new_tokens",
        "quantization",
    ):
        value = getattr(config, name, None)
        if value is not None:
            signature[name] = value
    return signature


def _research_request_key(
    router: Any,
    role: str,
    messages: Sequence[Mapping[str, Any]] | Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> str:
    return _sha256(
        {
            "schema_version": _CACHE_SCHEMA,
            "role_signature": _role_signature(router, role),
            "messages": messages,
            "args": args,
            "kwargs": dict(kwargs),
        }
    )


def _checkpoint_path(key: str) -> Path:
    directory = _checkpoint_root() / key[:2]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def _research_note_payload(raw: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    note = value.get("research_note")
    if isinstance(note, Mapping):
        return note
    # Keep direct-note compatibility for small focused adapters while production uses
    # the top-level research_note envelope enforced by _RESEARCH_NOTE_SCHEMA.
    return value


def _valid_research_response(raw: Any) -> bool:
    note = _research_note_payload(raw)
    if note is None:
        return False
    if not isinstance(note.get("domain_id"), str):
        return False
    if not isinstance(note.get("claims"), list):
        return False
    if not isinstance(note.get("gaps"), list):
        return False
    if not isinstance(note.get("next_queries"), list):
        return False
    return isinstance(note.get("sufficient"), bool)


def _read_checkpoint(key: str) -> str | None:
    path = _checkpoint_path(key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if (
        payload.get("schema_version") != _CACHE_SCHEMA
        or payload.get("request_sha256") != key
    ):
        return None
    raw = payload.get("response")
    if not _valid_research_response(raw):
        return None
    expected = str(payload.get("response_sha256", ""))
    actual = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()
    return str(raw) if expected == actual else None


def _write_checkpoint(key: str, raw: str) -> None:
    if not _valid_research_response(raw):
        return
    path = _checkpoint_path(key)
    payload = {
        "schema_version": _CACHE_SCHEMA,
        "request_sha256": key,
        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "response": raw,
    }
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _json_message_payload(messages: Any) -> Any:
    if not isinstance(messages, Sequence) or isinstance(
        messages,
        (str, bytes, bytearray),
    ):
        return None
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")) != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            continue
    return None


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _without_prior(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_prior(item)
            for key, item in value.items()
            if str(key)
            not in {
                "prior",
                "prior_note",
                "previous_note",
                "previous_reflection",
            }
        }
    if isinstance(value, list):
        return [_without_prior(item) for item in value]
    return value


def _document_synthesis_key(router: Any, role: str, messages: Any) -> str | None:
    payload = _json_message_payload(messages)
    if not isinstance(payload, Mapping):
        return None
    if not (
        _contains_key(payload, "evidence_document")
        and _contains_key(payload, "page_notes")
    ):
        return None
    return _sha256(
        {
            "schema": "mmm/document-synthesis-fixed-point-v1",
            "role_signature": _role_signature(router, role),
            "payload_without_prior": _without_prior(payload),
        }
    )


def _install_research_generation_resilience(model_router_module: Any) -> None:
    current = model_router_module.ModelRouter.generate_text
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate_text(
        self: Any,
        role: str,
        messages: Any,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        call_kwargs = dict(kwargs)
        is_research = str(call_kwargs.get("tool_stage", "")) == "research"
        synthesis_key = (
            _document_synthesis_key(self, role, messages) if is_research else None
        )

        # All evidence pages have already been materialized and read before this
        # synthesis. Re-entering search_project_rag/search_code_rag/external_mcp_call
        # here only repeats retrieval over unchanged evidence.
        if synthesis_key is not None:
            call_kwargs["enable_tools"] = False

        checkpointable = is_research and call_kwargs.get("enable_tools") is False
        request_key = (
            _research_request_key(self, role, messages, args, call_kwargs)
            if checkpointable
            else None
        )

        lock = _key_lock(request_key) if request_key is not None else None
        if lock is not None:
            lock.acquire()
        try:
            if request_key is not None:
                cached = _read_checkpoint(request_key)
                if cached is not None:
                    if synthesis_key is not None:
                        with _SYNTHESIS_LOCK:
                            _SYNTHESIS_RESULTS.setdefault(synthesis_key, cached)
                    return cached

            # An insufficient but schema-valid synthesis cannot gain new evidence after
            # tools are disabled. Replaying the first valid synthesis makes the existing
            # caller's fixed-point detector terminate without another backend generation.
            if synthesis_key is not None:
                with _SYNTHESIS_LOCK:
                    prior_result = _SYNTHESIS_RESULTS.get(synthesis_key)
                if prior_result is not None:
                    return prior_result

            raw = current(self, role, messages, *args, **call_kwargs)
            if request_key is not None and _valid_research_response(raw):
                _write_checkpoint(request_key, raw)
            if synthesis_key is not None and _valid_research_response(raw):
                with _SYNTHESIS_LOCK:
                    _SYNTHESIS_RESULTS.setdefault(synthesis_key, raw)
            return raw
        finally:
            if lock is not None:
                lock.release()

    setattr(generate_text, _MARKER, True)
    generate_text.__wrapped__ = current  # type: ignore[attr-defined]
    model_router_module.ModelRouter.generate_text = generate_text


def _managed_server_owned(autotune: Any) -> bool:
    managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "").rstrip("/")
    configured = (os.environ.get("LLAMA_SERVER_URL") or "").strip().rstrip("/")
    return bool(managed_url and configured == managed_url)


def _rearm_managed_server(autotune: Any, *, force: bool) -> bool:
    lock = getattr(autotune, "_AUTOTUNE_LOCK", None)
    if lock is None:
        return False
    with lock:
        process = getattr(autotune, "_MANAGED_PROCESS", None)
        if process is None:
            return False
        alive = process.poll() is None
        if alive and not force:
            return False
        old_key = getattr(autotune, "_MANAGED_KEY", None)
        old_url = getattr(autotune, "_MANAGED_URL", None)
        if alive and force:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=3.0)
                except Exception:
                    pass
        if old_key:
            attempted = getattr(autotune, "_ATTEMPTED_KEYS", None)
            if isinstance(attempted, set):
                attempted.discard(old_key)
        configured = (os.environ.get("LLAMA_SERVER_URL") or "").rstrip("/")
        if old_url and configured == str(old_url).rstrip("/"):
            os.environ.pop("LLAMA_SERVER_URL", None)
        autotune._MANAGED_PROCESS = None
        autotune._MANAGED_KEY = None
        autotune._MANAGED_URL = None
        return True


def _install_autotune_rearm(autotune: Any) -> None:
    current = autotune.ensure_tuned_server
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def ensure_tuned_server(config: Any, request: Any) -> str:
        _rearm_managed_server(autotune, force=False)
        return current(config, request)

    setattr(ensure_tuned_server, _MARKER, True)
    ensure_tuned_server.__wrapped__ = current  # type: ignore[attr-defined]
    autotune.ensure_tuned_server = ensure_tuned_server


def _transport_failure(exc: BaseException) -> bool:
    markers = (
        "connection refused",
        "connection reset",
        "broken pipe",
        "remote disconnected",
        "remote end closed connection",
        "server disconnected",
        "errno 111",
        "errno 104",
    )
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".lower()
        if any(marker in text for marker in markers):
            return True
        next_exc = current.__cause__ or current.__context__
        current = next_exc if isinstance(next_exc, BaseException) else None
    return False


def _install_managed_backend_recovery(llama_adapter_module: Any, autotune: Any) -> None:
    adapter_cls = llama_adapter_module.LlamaCppAdapter
    current = adapter_cls.generate
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def generate(self: Any, request: Any) -> Any:
        try:
            return current(self, request)
        except Exception as first_error:
            if not _transport_failure(first_error):
                raise
            if not _managed_server_owned(autotune):
                raise
            # Only an MMM-owned local server is eligible. External/user-provided
            # servers are never terminated or silently replaced. Replay the exact
            # same request object once after rearming the managed process.
            if not _rearm_managed_server(autotune, force=True):
                raise
            try:
                return current(self, request)
            except Exception as retry_error:
                raise retry_error from first_error

    setattr(generate, _MARKER, True)
    generate.__wrapped__ = current  # type: ignore[attr-defined]
    adapter_cls.generate = generate


def install() -> None:
    from . import llama_server_autotune, model_router
    from .model_adapters import llama_cpp_adapter

    _install_research_generation_resilience(model_router)
    _install_autotune_rearm(llama_server_autotune)
    _install_managed_backend_recovery(llama_cpp_adapter, llama_server_autotune)
    globals()["_mmm_long_run_resilience_installed"] = _MARKER
