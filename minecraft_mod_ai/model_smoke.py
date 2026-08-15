from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .model_adapters import ModelBackendError
from .model_router import ModelRouter
from .model_registry import ModelRegistry


def run_model_smoke(
    *,
    role: str,
    profile: str,
    output_dir: str | Path,
    media_path: str | Path | None = None,
    audio_path: str | Path | None = None,
) -> dict[str, Any]:
    """Actually load and exercise one configured role, then record measurements."""

    router = ModelRouter(profile=profile)
    config = ModelRegistry().role(profile, role)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    before = _cuda_snapshot()
    started = time.perf_counter()
    output: Any
    output_file: Path | None = None
    try:
        if role == "embedding":
            vectors = router.embed(["approved-target Fabric register a custom item"])
            output = {"dimensions": len(vectors[0]), "norm": sum(v * v for v in vectors[0]) ** 0.5}
        elif role == "reranker":
            output = {
                "scores": router.rerank(
                    "approved-target Fabric item registration",
                    [
                        "approved-target Fabric Registry.register example",
                        "Forge 1.21 DeferredRegister example",
                    ],
                )
            }
        elif role == "image_generator":
            output_file = target / "smoke-image.png"
            router.generate_image(
                role,
                prompt="original blue crystal Minecraft inventory icon, no text, no watermark",
                output_path=output_file,
                width=512,
                height=512,
                seed=17,
            )
            output = {
                "path": str(output_file),
                "sha256": _sha256(output_file),
            }
        elif role == "speech_recognition":
            if audio_path is None:
                raise ValueError("speech_recognition smoke test requires audio_path.")
            output = {"transcript": router.transcribe(role, audio_path)}
        else:
            media = [media_path] if media_path is not None else []
            text = router.generate_text(
                role,
                [
                    {
                        "role": "system",
                        "content": "Reply with exactly one JSON object and no markdown.",
                    },
                    {
                        "role": "user",
                        "content": (
                            '{"status":"ok","role":"' + role + '","target":"approved-target Fabric"}'
                        ),
                    },
                ],
                media_paths=media,
                response_format="json",
            )
            output = {"text": text}
        elapsed = time.perf_counter() - started
        after = _cuda_snapshot()
        report = {
            "schema_version": "mmm/model-smoke-v1",
            "profile": profile,
            "role": role,
            "model_id": config.model_id,
            "adapter": config.adapter,
            "elapsed_seconds": round(elapsed, 4),
            "cuda_before": before,
            "cuda_after": after,
            "output": output,
            "status": "PASS",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        report = {
            "schema_version": "mmm/model-smoke-v1",
            "profile": profile,
            "role": role,
            "model_id": config.model_id,
            "adapter": config.adapter,
            "elapsed_seconds": round(elapsed, 4),
            "cuda_before": before,
            "cuda_after": _cuda_snapshot(),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ModelBackendError):
            report["backend_cause"] = str(exc.cause)
    report_path = target / f"{role}-smoke.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    report["report_sha256"] = _sha256(report_path)
    return report


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "free_mb": int(free / (1024 * 1024)),
            "total_mb": int(total / (1024 * 1024)),
            "allocated_mb": int(torch.cuda.memory_allocated() / (1024 * 1024)),
            "reserved_mb": int(torch.cuda.memory_reserved() / (1024 * 1024)),
            "peak_allocated_mb": int(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            ),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
