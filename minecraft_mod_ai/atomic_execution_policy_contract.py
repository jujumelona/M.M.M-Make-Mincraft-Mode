from __future__ import annotations

import hashlib
from functools import wraps
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def install(atomic_module: Any, orchestrator_module: Any) -> None:
    """Order immutable input checks before IR and reserve strict IR for binaries."""

    cls = orchestrator_module.CompleteProductionOrchestrator
    current = cls.execute
    if getattr(current, "_mmm_atomic_execution_policy", False):
        return
    base = getattr(current, "__wrapped__", current)

    @wraps(base)
    def execute(self: Any, proposal: Any, *args: Any, **kwargs: Any):
        parsed = (
            proposal
            if isinstance(proposal, orchestrator_module.CompleteProposal)
            else orchestrator_module.CompleteProposal.from_dict(proposal)
        )
        existing_input = kwargs.get("existing_input")
        bound = str(getattr(parsed, "existing_input_sha256", ""))
        supplied = existing_input is not None
        if bool(bound) != supplied:
            if bound:
                raise orchestrator_module.CompleteProductionError(
                    "This approved complete plan is bound to an existing-project "
                    "ZIP, so the same ZIP is required."
                )
            raise orchestrator_module.CompleteProductionError(
                "An existing-project ZIP may be used only with a complete plan "
                "that was approved with that input."
            )
        if bound and existing_input is not None:
            path = Path(existing_input).expanduser().resolve()
            if not path.is_file() or path.is_symlink():
                raise orchestrator_module.CompleteProductionError(
                    "The approved existing-project ZIP is missing or unsafe."
                )
            if _sha256_file(path) != bound:
                raise orchestrator_module.CompleteProductionError(
                    "The existing-project ZIP changed after complete-plan approval."
                )

        options = kwargs.get("options")
        source_only = bool(getattr(options, "source_only", False))
        design = getattr(parsed, "game_design", {})
        ir = (
            design.get("_atomic_requirement_ir")
            if isinstance(design, dict)
            else None
        )
        # Source packages are non-release artifacts. Legacy/manual source-only
        # workflows may omit the new IR, but if they carry one it must be valid.
        # Any binary build path requires a complete current IR.
        if ir is not None or not source_only:
            try:
                atomic_module.validate_ir(parsed)
            except atomic_module.AtomicRequirementError as exc:
                raise orchestrator_module.CompleteProductionError(str(exc)) from exc
        return base(self, parsed, *args, **kwargs)

    execute._mmm_atomic_release_guard = True
    execute._mmm_atomic_execution_policy = True
    cls.execute = execute
