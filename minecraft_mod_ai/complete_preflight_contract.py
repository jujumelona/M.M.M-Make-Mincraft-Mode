from __future__ import annotations

"""Preflight policy for approved complete-production execution."""

from pathlib import Path
import tempfile
from typing import Any

from .complete_orchestrator_support import CompleteProductionError

REQUIRED_GATE_TO_EVIDENCE = {
    "registry": "source",
    "resource": "source",
    "recipe": "source",
    "source static validation": "source",
    "generated resource validation": "source",
    "jdt": "jdt",
    "jdt diagnostics": "jdt",
    "gradle": "gradle",
    "gradle clean build": "gradle",
    "target compile": "gradle",
    "gametest": "gametest",
    "gametest spawn and attributes": "gametest",
    "worldgen runtime validation": "gametest",
    "jar": "jar",
    "jar validation": "jar",
    "runtime": "runtime_client",
    "minecraft server client runtime": "runtime_client",
    "network protocol validation": "playtest",
    "mineflayer playtest": "playtest",
    "runtime interaction tests": "playtest",
    "runtime animation review": "runtime_visual",
    "blockbench uv and bone hierarchy review": "blockbench",
    "blockbench uv render review": "blockbench",
    "visual review": "visual",
    "client gui and validated network action test": "playtest_visual",
    "research ledger integrity": "research_ledger",
}


def normalize_required_gate(value: str) -> str:
    return " ".join(
        "".join(
            character.casefold() if character.isalnum() else " "
            for character in value
        ).split()
    )


def validate_required_gate_contract(proposal: Any) -> None:
    unsupported: list[str] = []
    for module in getattr(proposal, "modules", ()):
        module_id = str(getattr(module, "module_id", "") or "")
        for gate in getattr(module, "required_gates", ()):
            rendered = str(gate).strip()
            if not rendered:
                continue
            if normalize_required_gate(rendered) not in REQUIRED_GATE_TO_EVIDENCE:
                unsupported.append(f"{module_id}:{rendered}")
    if unsupported:
        raise CompleteProductionError(
            "Approved proposal contains unsupported required gates: "
            + ", ".join(sorted(unsupported))
        )


def validate_platform_toolchain_preflight(
    platform: Any,
    options: Any,
) -> None:
    """Prove one already-bound target toolchain before model/generation work."""

    source_only = bool(getattr(options, "source_only", False))
    run_jdt = bool(getattr(options, "run_jdt", False))
    if source_only and not run_jdt:
        return

    try:
        from .platform_catalog import adapter_for_lock_values

        adapter = adapter_for_lock_values(platform)
        raw_java = getattr(adapter, "java_version")
    except Exception as exc:
        raise CompleteProductionError(
            "Approved target toolchain cannot be resolved before execution: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    from .java_lsp import (
        _java_major_version,
        _parse_java_major,
        _resolve_project_java_home,
    )

    required = _parse_java_major(str(raw_java))
    if required is None or required <= 0:
        raise CompleteProductionError(
            f"Approved platform has invalid java_version={raw_java!r}."
        )

    try:
        project_home = _resolve_project_java_home(
            required,
            require_compiler=True,
        )
        actual = _java_major_version(project_home)
    except Exception as exc:
        raise CompleteProductionError(
            "Java toolchain preflight failed before generation: "
            f"required={required}; {type(exc).__name__}: {exc}"
        ) from exc
    if actual != required:
        raise CompleteProductionError(
            "Java toolchain preflight resolved the wrong JDK: "
            f"required={required}, actual={actual}, home={project_home}"
        )

    if not source_only:
        try:
            from .runner import GradleRunner, production_gradle_cache_dir

            GradleRunner(
                production_gradle_cache_dir(),
                download_timeout_seconds=300,
            ).ensure_gradle(
                str(adapter.gradle),
                str(adapter.gradle_sha256),
                lock_timeout_seconds=360,
            )
        except Exception as exc:
            raise CompleteProductionError(
                "Gradle distribution preflight failed before generation: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    if not run_jdt:
        return

    owner_major = max(21, required)
    try:
        owner_home = _resolve_project_java_home(
            owner_major,
            require_compiler=True,
        )
        from .jvm_owner_bootstrap import owner_command

        with tempfile.TemporaryDirectory(prefix="mmm-jdt-preflight-") as raw:
            command = owner_command(
                Path(raw) / "owner",
                timeout_seconds=300,
                java_home=owner_home,
                required_major=owner_major,
            )
        if not command or not Path(command[0]).is_file():
            raise RuntimeError("JDT owner bootstrap returned no executable Java command")
    except Exception as exc:
        raise CompleteProductionError(
            "JDT owner preflight failed before generation: "
            f"project_release={required}, owner_java={owner_major}; "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _validate_java_toolchain_preflight(
    proposal: Any,
    options: Any,
) -> None:
    try:
        platform = proposal.base_proposal.spec.platform
    except (AttributeError, TypeError) as exc:
        raise CompleteProductionError(
            "Approved proposal is missing the locked Java toolchain."
        ) from exc
    validate_platform_toolchain_preflight(platform, options)


def validate_external_execution_preflight(
    proposal: Any,
    options: Any,
) -> None:
    # Executable CompleteProposal instances carry a locked base_proposal.
    # Minimal policy-only projections (used by host policy callers/tests)
    # have only external_runtime_required: they must remain valid inputs to
    # this *runtime-policy* check rather than crashing on a missing toolchain.
    # Actual production always passes the validated full proposal.
    if getattr(proposal, "base_proposal", None) is not None:
        _validate_java_toolchain_preflight(proposal, options)

    if bool(getattr(options, "source_only", False)):
        return

    required = bool(getattr(proposal, "external_runtime_required", False))
    if required:
        disabled = [
            name
            for name, enabled in (
                ("runtime", getattr(options, "run_runtime", False)),
                ("client", getattr(options, "run_client", False)),
                ("mineflayer", getattr(options, "run_mineflayer", False)),
                ("visual-review", getattr(options, "run_visual_review", False)),
            )
            if not enabled
        ]
        if disabled:
            raise CompleteProductionError(
                "Approved proposal requires external runtime verification, but "
                "these checks are disabled: " + ", ".join(disabled)
            )

    if bool(getattr(options, "run_client", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Client verification requires runtime verification."
        )

    has_entity_review = any(
        getattr(module, "kind", "") in {"entity", "boss", "npc"}
        for module in getattr(proposal, "modules", ())
    )
    if (
        has_entity_review
        and not bool(getattr(options, "source_only", False))
        and not bool(getattr(options, "run_blockbench", False))
    ):
        raise CompleteProductionError(
            "Entity production requires Blockbench UV/render verification."
        )

    if bool(getattr(options, "run_runtime", False)):
        if not bool(getattr(options, "eula_accepted", False)):
            raise CompleteProductionError(
                "Runtime verification was requested without explicit "
                "Minecraft EULA acceptance."
            )
        raw_launcher = getattr(options, "server_launcher", None)
        if not isinstance(raw_launcher, str) or not raw_launcher.strip():
            raise CompleteProductionError(
                "Runtime verification requires server_launcher before "
                "generation starts."
            )
        launcher = Path(raw_launcher).expanduser().resolve()
        if not launcher.is_file() or launcher.is_symlink():
            raise CompleteProductionError(
                "server_launcher must be an existing regular file before "
                "generation starts."
            )

    if bool(getattr(options, "run_mineflayer", False)):
        if not bool(getattr(options, "run_runtime", False)):
            raise CompleteProductionError(
                "Mineflayer verification requires runtime verification."
            )
        actions = getattr(options, "playtest_actions", ())
        if not isinstance(actions, (list, tuple)) or not actions:
            raise CompleteProductionError(
                "Mineflayer verification requires explicit playtest_actions "
                "before generation starts."
            )
        expected_tests = tuple(
            str(value)
            for value in getattr(proposal, "acceptance_tests", ())
        )
        if expected_tests:
            expected_set = set(expected_tests)
            covered: set[str] = set()
            unknown: set[str] = set()
            for action in actions:
                if not isinstance(action, dict):
                    continue
                if str(action.get("action") or "") != "wait_for":
                    continue
                raw_test = action.get("acceptance_test")
                if raw_test is None:
                    continue
                test = str(raw_test)
                if test in expected_set:
                    covered.add(test)
                else:
                    unknown.add(test)
            missing = [test for test in expected_tests if test not in covered]
            if missing or unknown:
                details: list[str] = []
                if missing:
                    details.append("missing=" + ", ".join(missing))
                if unknown:
                    details.append("unknown=" + ", ".join(sorted(unknown)))
                raise CompleteProductionError(
                    "Mineflayer playtest actions do not match the approved "
                    "acceptance tests: " + "; ".join(details)
                )

    if bool(getattr(options, "run_visual_review", False)) and not bool(
        getattr(options, "run_runtime", False)
    ):
        raise CompleteProductionError(
            "Visual verification requires the disposable runtime."
        )


__all__ = [
    "REQUIRED_GATE_TO_EVIDENCE",
    "normalize_required_gate",
    "validate_external_execution_preflight",
    "validate_platform_toolchain_preflight",
    "validate_required_gate_contract",
]
