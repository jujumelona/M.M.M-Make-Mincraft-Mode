from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .model_adapters import AdapterConfig, ModelBackendError
from .model_registry import ModelRegistry
from .model_router import ModelRouter

_DEFAULT_ROLES = ("planner", "researcher", "coder", "coder_safe", "visual_critic")


class _SmokeRegistry(ModelRegistry):
    """Registry view that caps decode length without changing production config."""

    def __init__(self, *, max_new_tokens: int) -> None:
        super().__init__()
        self._smoke_max_new_tokens = max(1, int(max_new_tokens))

    def role(self, profile: str, role: str) -> AdapterConfig:
        config = super().role(profile, role)
        return replace(
            config,
            max_new_tokens=min(config.max_new_tokens, self._smoke_max_new_tokens),
        )


def _backend_identity(config: AdapterConfig) -> tuple[str, ...]:
    extra = config.extra if isinstance(config.extra, Mapping) else {}
    return (
        str(config.provider),
        str(config.adapter),
        str(config.model_id),
        str(config.quantization or ""),
        str(extra.get("gguf_filename", "")),
        str(extra.get("mmproj_filename", "")),
    )


def run_model_smoke(
    *,
    role: str,
    profile: str,
    output_dir: str | Path,
    media_path: str | Path | None = None,
    registry: ModelRegistry | None = None,
    covered_roles: Sequence[str] = (),
) -> dict[str, Any]:
    """Actually load and minimally exercise one configured backend."""

    active_registry = registry or ModelRegistry()
    router = ModelRouter(profile=profile, registry=active_registry)
    config = active_registry.role(profile, role)
    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    before = _cuda_snapshot()
    started = time.perf_counter()
    output: Any
    try:
        if role == "embedding":
            vectors = router.embed(["Fabric Registry register custom item"])
            output = {
                "dimensions": len(vectors[0]),
                "norm": sum(v * v for v in vectors[0]) ** 0.5,
            }
        elif role == "reranker":
            output = {
                "scores": router.rerank(
                    "Fabric item registration",
                    ["Fabric Registry.register example", "unrelated recipe example"],
                )
            }
        elif role == "image_generator":
            output_file = target / "smoke-image.png"
            router.generate_image(
                role,
                prompt="blue crystal Minecraft inventory icon",
                output_path=output_file,
                width=256,
                height=256,
                seed=17,
            )
            output = {"path": str(output_file), "sha256": _sha256(output_file)}
        else:
            media = [media_path] if media_path is not None else []
            with router.generation_session(role):
                text = router.generate_text(
                    role,
                    [
                        {
                            "role": "system",
                            "content": "Return exactly one valid JSON object. No markdown.",
                        },
                        {"role": "user", "content": "Return exactly {}."},
                    ],
                    media_paths=media,
                    response_format="json",
                    enable_tools=False,
                )
            if not str(text).strip():
                raise RuntimeError("model returned an empty smoke response")
            output = {"text": text}
        elapsed = time.perf_counter() - started
        report = {
            "schema_version": "mmm/model-smoke-v2",
            "profile": profile,
            "role": role,
            "covered_roles": list(covered_roles or (role,)),
            "model_id": config.model_id,
            "adapter": config.adapter,
            "elapsed_seconds": round(elapsed, 4),
            "cuda_before": before,
            "cuda_after": _cuda_snapshot(),
            "output": output,
            "status": "PASS",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        report = {
            "schema_version": "mmm/model-smoke-v2",
            "profile": profile,
            "role": role,
            "covered_roles": list(covered_roles or (role,)),
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


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"model smoke exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_profile_smoke(
    *,
    profile: str,
    roles: Sequence[str],
    output_dir: str | Path,
    max_new_tokens: int = 4,
    timeout: int = 120,
    quick: bool = False,
    require_real: bool = False,
    expected_adapters: Sequence[str] = (),
) -> dict[str, Any]:
    """Exercise requested roles, deduping identical backends in quick mode."""

    requested = tuple(dict.fromkeys(role.strip() for role in roles if role.strip()))
    if not requested:
        raise ValueError("at least one smoke role is required")

    registry = _SmokeRegistry(max_new_tokens=max_new_tokens)
    configs = {role: registry.role(profile, role) for role in requested}

    if expected_adapters:
        expected = tuple(expected_adapters)
        if len(expected) != len(requested):
            raise ValueError("--expected-adapters must contain one value per requested role")
        mismatches = [
            f"{role}: expected {wanted}, got {configs[role].adapter}"
            for role, wanted in zip(requested, expected)
            if wanted != configs[role].adapter
        ]
        if mismatches:
            raise ValueError("; ".join(mismatches))

    if require_real:
        mock_roles = [
            role
            for role, config in configs.items()
            if config.adapter == "mock" or config.model_id.startswith("mock/")
        ]
        if mock_roles:
            raise ValueError(
                "real-model smoke selected mock roles: " + ", ".join(mock_roles)
            )

    if quick:
        grouped: dict[tuple[str, ...], list[str]] = {}
        for role in requested:
            grouped.setdefault(_backend_identity(configs[role]), []).append(role)
        groups = [(members[0], tuple(members)) for members in grouped.values()]
    else:
        groups = [(role, (role,)) for role in requested]

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for representative, covered in groups:
        with _deadline(timeout):
            reports.append(
                run_model_smoke(
                    role=representative,
                    profile=profile,
                    output_dir=target,
                    registry=registry,
                    covered_roles=covered,
                )
            )

    failures = [report for report in reports if report["status"] != "PASS"]
    summary = {
        "schema_version": "mmm/model-smoke-suite-v2",
        "profile": profile,
        "requested_roles": list(requested),
        "quick": bool(quick),
        "unique_backends_exercised": len(reports),
        "max_new_tokens": max_new_tokens,
        "timeout_seconds_per_backend": timeout,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "status": "FAIL" if failures else "PASS",
        "checks": reports,
    }
    (target / "MODEL_SMOKE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


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
            "peak_allocated_mb": int(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load and minimally exercise real model backends from the registry."
    )
    parser.add_argument("--profile", default="t4_quality")
    parser.add_argument("--roles", default=",".join(_DEFAULT_ROLES))
    parser.add_argument("--output-dir", default="audit")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--require-real", action="store_true")
    parser.add_argument("--expected-adapters", default="")
    args = parser.parse_args(argv)

    try:
        summary = run_profile_smoke(
            profile=args.profile,
            roles=_csv(args.roles),
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
            quick=args.quick,
            require_real=args.require_real,
            expected_adapters=_csv(args.expected_adapters),
        )
    except Exception as exc:
        summary = {
            "schema_version": "mmm/model-smoke-suite-v2",
            "profile": args.profile,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        target = Path(args.output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        (target / "MODEL_SMOKE.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
