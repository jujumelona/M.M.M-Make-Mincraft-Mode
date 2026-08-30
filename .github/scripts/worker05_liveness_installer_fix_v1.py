from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "minecraft_mod_ai/llama_completion_liveness_contract.py"
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    text = text.replace(old, new, 1)


replace_once(
    '''def _install_raw_client_watchdog(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current_init = client_type.__init__\n    if not getattr(current_init, _CLIENT_INIT_MARKER, False):\n\n        @wraps(current_init)\n        def progress_checked_init(self: Any, client: Any) -> None:\n            current_init(self, _wrap_raw_client(client, stream_module))\n\n        setattr(progress_checked_init, _CLIENT_INIT_MARKER, True)\n        progress_checked_init.__wrapped__ = current_init  # type: ignore[attr-defined]\n        client_type.__init__ = progress_checked_init\n''',
    '''def _install_raw_client_watchdog(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current_init = client_type.__init__\n    if not getattr(current_init, _CLIENT_INIT_MARKER, False):\n\n        @wraps(current_init)\n        def progress_checked_init(self: Any, *args: Any, **kwargs: Any) -> None:\n            if args:\n                updated_args = list(args)\n                updated_args[0] = _wrap_raw_client(updated_args[0], stream_module)\n                current_init(self, *updated_args, **kwargs)\n                return\n            if "client" in kwargs:\n                updated_kwargs = dict(kwargs)\n                updated_kwargs["client"] = _wrap_raw_client(\n                    updated_kwargs["client"], stream_module\n                )\n                current_init(self, **updated_kwargs)\n                return\n            current_init(self, *args, **kwargs)\n\n        setattr(progress_checked_init, _CLIENT_INIT_MARKER, True)\n        progress_checked_init.__wrapped__ = current_init  # type: ignore[attr-defined]\n        client_type.__init__ = progress_checked_init\n''',
    "raw client watchdog constructor",
)

replace_once(
    '''def _install_stream_progress_payload(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current = client_type.stream\n    if getattr(current, _STREAM_MARKER, False):\n        return\n''',
    '''def _install_stream_progress_payload(stream_module: Any) -> None:\n    client_type = stream_module._StreamingCompletionClient\n    current = getattr(client_type, "stream", None)\n    if not callable(current) or getattr(current, _STREAM_MARKER, False):\n        return\n''',
    "optional stream-progress installer",
)

path.write_text(text, encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
