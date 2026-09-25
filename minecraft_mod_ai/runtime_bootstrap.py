from __future__ import annotations

"""Canonical one-time installation of runtime validation/generation contracts."""

import threading

_LOCK = threading.RLock()
_BOOTSTRAPPED = False


def bootstrap_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    with _LOCK:
        if _BOOTSTRAPPED:
            return

        from . import (
            extended_content_generator,
            java_lsp,
            repair_engine,
            runner,
            validation_execution_contract,
        )
        from . import extended_registration_contract
        from . import java_lsp_process_safety_contract
        from . import research_validation_fingerprint_performance
        from . import runner_parallel_validation_contract

        java_lsp_process_safety_contract.install(java_lsp)
        runner_parallel_validation_contract.install(
            runner_module=runner,
            validation_module=validation_execution_contract,
        )
        validation_execution_contract.install(
            runner,
            java_lsp,
            repair_engine,
        )
        research_validation_fingerprint_performance.harden(
            validation_execution_contract
        )
        extended_registration_contract.install(extended_content_generator)

        _BOOTSTRAPPED = True


__all__ = ["bootstrap_runtime"]
