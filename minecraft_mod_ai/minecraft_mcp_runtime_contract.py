from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any

from .external_mcp_router import ExternalMCPRouter


def install(orchestrator_module: Any) -> None:
    cls = orchestrator_module.CompleteProductionOrchestrator
    original = cls._run_playtest
    if getattr(original, "_mmm_external_mcp_runtime", False):
        return

    @wraps(original)
    def run_playtest(actions):
        base = original(actions)
        try:
            from .platform_runtime_contract import _ACTIVE_ADAPTER

            adapter = _ACTIVE_ADAPTER.get()
        except Exception:
            adapter = None
        if adapter is None:
            return base

        capabilities = (
            ("runtime_inspection", {}),
            ("runtime_visual", {}),
            ("runtime_world_read", {}),
        )
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(capabilities)) as pool:
            futures = {
                pool.submit(
                    _runtime_read,
                    capability,
                    arguments,
                    adapter,
                ): capability
                for capability, arguments in capabilities
            }
            for future in as_completed(futures):
                capability = futures[future]
                try:
                    bundle = future.result()
                except Exception as exc:
                    bundle = {
                        "schema_version": "mmm/external-mcp-evidence-bundle-v1",
                        "capability": capability,
                        "stage": "runtime",
                        "status": "UNAVAILABLE",
                        "evidence": [],
                        "attempts": [
                            {
                                "status": "ROUTER_ERROR",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                    }
                results.append(
                    {
                        "capability": capability,
                        "status": bundle.get("status", "UNAVAILABLE"),
                        "bundle_sha256": bundle.get("bundle_sha256", ""),
                        "attempts": bundle.get("attempts", []),
                        "evidence": [
                            {
                                key: receipt.get(key)
                                for key in (
                                    "server",
                                    "tool",
                                    "trust",
                                    "requested_target",
                                    "server_info",
                                    "result_sha256",
                                    "status",
                                )
                            }
                            for receipt in bundle.get("evidence", [])
                            if isinstance(receipt, dict)
                        ],
                    }
                )
        results.sort(key=lambda item: item["capability"])
        return {
            **base,
            "external_mcp": {
                "schema_version": "mmm/runtime-mcp-evidence-v1",
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "read_only": True,
                "test_harness_only": True,
                "routes": results,
            },
        }

    run_playtest._mmm_external_mcp_runtime = True
    cls._run_playtest = staticmethod(run_playtest)


def _runtime_read(capability: str, arguments: dict[str, Any], adapter: Any) -> dict[str, Any]:
    return ExternalMCPRouter(timeout_seconds=30.0).invoke(
        capability,
        stage="runtime",
        arguments=arguments,
        target=adapter,
        max_access="read",
        required=False,
        disposable_runtime=True,
    )
