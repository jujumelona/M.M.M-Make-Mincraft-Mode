from __future__ import annotations

from pathlib import Path

from .base import ModelBackendError, ModelConfigurationError, preflight_cuda, require_package, torch_dtype


class SpeechAdapter:
    def __init__(self, config) -> None:
        self.config = config

    def transcribe(self, audio_path: Path) -> str:
        cfg = self.config
        try:
            require_package("transformers", minimum="4.57.0")
            preflight_cuda(cfg)
            import torch
            from transformers import pipeline

            source = audio_path.expanduser().resolve()
            if not source.is_file():
                raise ModelConfigurationError(f"Audio file does not exist: {source}")
            device = 0 if torch.cuda.is_available() else -1
            recognizer = pipeline(
                "automatic-speech-recognition",
                model=cfg.model_id,
                device=device,
                torch_dtype=(torch_dtype(cfg.torch_dtype) if device >= 0 else torch.float32),
            )
            try:
                result = recognizer(str(source), return_timestamps=True)
                text = result.get("text") if isinstance(result, dict) else None
                if not isinstance(text, str):
                    raise ModelConfigurationError("Speech backend returned no transcript.")
                return text.strip()
            finally:
                del recognizer
        except ModelBackendError:
            raise
        except Exception as exc:
            raise ModelBackendError(
                role=cfg.role, model_id=cfg.model_id, cause=exc
            ) from exc
