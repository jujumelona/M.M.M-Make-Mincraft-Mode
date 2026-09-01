from __future__ import annotations

"""Runtime hardening for provenance and platform target selection.

This module intentionally patches narrow failure boundaries without weakening strict
validators or changing public APIs. It is installed before the package API surface is
imported so already-bound internal references can be updated consistently.
"""

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_INSTALLED = False


def _base_evidence_ref(value: Any) -> str:
    text = str(value or "").strip()
    marker = "#synthesis-"
    if marker in text:
        text = text.split(marker, 1)[0]
    return text


def _allowed_evidence_refs(group: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        ref = _base_evidence_ref(value)
        if ref and ref not in seen:
            seen.add(ref)
            refs.append(ref)

    for note in group:
        add(note.get("page_ref"))
        fragment = note.get("evidence_fragment")
        if isinstance(fragment, Mapping):
            add(fragment.get("page_ref"))
        for claim in note.get("claims", ()):
            if not isinstance(claim, Mapping):
                continue
            for ref in claim.get("evidence_refs", ()):
                add(ref)
        for procedure in note.get("procedures", ()):
            if not isinstance(procedure, Mapping):
                continue
            for ref in procedure.get("evidence_refs", ()):
                add(ref)
    return tuple(refs)


def _repair_note_provenance(
    note: Mapping[str, Any],
    *,
    allowed_refs: Sequence[str],
) -> dict[str, Any]:
    """Keep only host-issued provenance; never invent a citation for a claim."""

    result = dict(note)
    allowed = tuple(
        dict.fromkeys(
            _base_evidence_ref(ref)
            for ref in allowed_refs
            if _base_evidence_ref(ref)
        )
    )
    allowed_set = set(allowed)

    claims: list[Any] = []
    dropped = 0
    for claim in result.get("claims", ()):
        if not isinstance(claim, Mapping):
            dropped += 1
            continue
        item = dict(claim)
        refs = list(
            dict.fromkeys(
                _base_evidence_ref(ref)
                for ref in item.get("evidence_refs", ())
                if _base_evidence_ref(ref) in allowed_set
            )
        )
        if not refs:
            dropped += 1
            continue
        item["evidence_refs"] = refs
        claims.append(item)
    if "claims" in result:
        result["claims"] = claims

    procedures: list[Any] = []
    for procedure in result.get("procedures", ()):
        if not isinstance(procedure, Mapping):
            continue
        item = dict(procedure)
        if "evidence_refs" in item:
            refs = list(
                dict.fromkeys(
                    _base_evidence_ref(ref)
                    for ref in item.get("evidence_refs", ())
                    if _base_evidence_ref(ref) in allowed_set
                )
            )
            if not refs:
                continue
            item["evidence_refs"] = refs
        procedures.append(item)
    if "procedures" in result:
        result["procedures"] = procedures

    if dropped:
        gaps = [str(value) for value in result.get("gaps", ()) if str(value).strip()]
        gaps.append(
            f"{dropped} synthesized claim(s) were omitted because no host-issued "
            "evidence reference survived provenance validation."
        )
        result["gaps"] = gaps
        if not claims:
            result["sufficient"] = False
    return result




@dataclass(frozen=True)
class _TargetProbe:
    loader: str
    minecraft_version: str
    deterministic_module_kinds: frozenset[str] = frozenset()
    edition: str = "java"
    yarn_mappings: str = "mojang"
    mappings_kind: str = "mojang"
    mappings_version: str = "mojang"

    @property
    def adapter_id(self) -> str:
        return f"probe:{self.loader}:{self.minecraft_version}"


def _version_key(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(value).split("."):
        try:
            parts.append(int(token))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _install_machine_only_pack_metadata() -> None:
    """Keep the canonical cached pack resolver intact.

    ``platform_live_discovery._official_pack_versions`` already uses Mojang's machine
    metadata as the primary source, preserves a bounded official fallback, returns the
    three-value public contract, and exposes ``lru_cache`` helpers such as
    ``cache_clear``. Replacing it here used to discard all of those guarantees.
    """

    return


def _install_two_stage_platform_optimizer() -> None:
    from . import platform_optimizer as optimizer

    original = optimizer.optimize_platform
    if getattr(original, "_mmm_two_stage", False):
        return

    def hardened_optimize_platform(
        prompt: str,
        *,
        design: Mapping[str, Any] | None = None,
        module_kinds: Any = (),
        loader_constraint: str | None = None,
        version_constraint: str | None = None,
        top_k: int = 4,
        discovery_client: Any | None = None,
        target_research_fn: Any | None = None,
        search_fn: Any | None = None,
        version_fn: Any | None = None,
    ) -> Any:
        # Deterministic fixture hooks retain the original path exactly.
        if search_fn is not None or version_fn is not None:
            return original(
                prompt,
                design=design,
                module_kinds=module_kinds,
                loader_constraint=loader_constraint,
                version_constraint=version_constraint,
                top_k=top_k,
                discovery_client=discovery_client,
                target_research_fn=target_research_fn,
                search_fn=search_fn,
                version_fn=version_fn,
            )

        queries = optimizer.capability_queries(
            prompt,
            design=design,
            module_kinds=module_kinds,
        )
        diagnostics: list[str] = []
        target_keys = optimizer.discover_target_keys(
            loader=loader_constraint,
            minecraft_version=version_constraint,
            limit_per_loader=12,
            diagnostics=diagnostics,
        )
        if not target_keys:
            target = "/".join(
                value for value in (version_constraint, loader_constraint) if value
            ) or "automatic"
            detail = "; ".join(diagnostics) or "no provider-discovered target was returned"
            raise ValueError(
                f"No executable platform provider can satisfy target {target!r}. Diagnostics: {detail}"
            )

        discovery_mode = os.environ.get("MMM_ECOSYSTEM_DISCOVERY", "auto").strip().lower()
        if discovery_mode not in {"auto", "on", "off"}:
            raise ValueError("MMM_ECOSYSTEM_DISCOVERY must be auto, on or off.")
        if discovery_mode == "off":
            if len(target_keys) != 1:
                raise ValueError(
                    "Ecosystem discovery is disabled and multiple executable platform targets remain. "
                    "Supply an explicit Minecraft target or enable discovery."
                )
            loader, version = target_keys[0]
            return original(
                prompt,
                design=design,
                module_kinds=module_kinds,
                loader_constraint=loader,
                version_constraint=version,
                top_k=1,
                discovery_client=discovery_client,
                target_research_fn=target_research_fn,
            )

        probes = tuple(
            _TargetProbe(loader=loader, minecraft_version=version)
            for loader, version in target_keys
        )
        client = discovery_client or optimizer.EcosystemDiscoveryClient()
        neutral, neutral_errors = optimizer._parallel_neutral_shallow(queries, client)
        shallow_count = sum(len(value) for value in neutral.values())
        matrix, matrix_errors = optimizer._parallel_support_matrix(probes, queries, client)

        selected_probe = max(
            probes,
            key=lambda probe: (
                optimizer._support_score(
                    probe,
                    queries,
                    matrix.get(probe.adapter_id, {}),
                ),
                _version_key(probe.minecraft_version),
                probe.loader,
            ),
        )

        # Expensive provider resolution happens exactly once, after the lightweight
        # support matrix chooses a target. A transport/source failure is surfaced;
        # it must never be reinterpreted as evidence that an older version is better.
        try:
            adapter = optimizer.adapter_for_target(
                selected_probe.minecraft_version,
                selected_probe.loader,
            )
        except Exception as exc:  # noqa: BLE001 - preserve failure class in diagnostic
            raise ValueError(
                "Selected platform target metadata is unavailable; refusing implicit "
                f"version downgrade for {selected_probe.minecraft_version}/{selected_probe.loader}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        selected_matrix = {
            adapter.adapter_id: dict(matrix.get(selected_probe.adapter_id, {}))
        }
        deep = optimizer._parallel_deep(
            (adapter,),
            queries=queries,
            matrix=selected_matrix,
            client=client,
            target_research_fn=target_research_fn,
            inherited_errors=(*diagnostics, *neutral_errors, *matrix_errors),
            shallow_candidate_count=shallow_count,
        )
        if not deep:
            raise ValueError("No executable platform target survived evidence verification.")
        evidence = deep[0]
        return optimizer.PlatformOptimization(
            selected=adapter,
            evidence=evidence,
            candidates=(evidence,),
            capability_queries=queries,
            discovery_mode="lightweight-support-matrix_then-single-target-full-resolution",
        )

    hardened_optimize_platform._mmm_two_stage = True  # type: ignore[attr-defined]
    optimizer.optimize_platform = hardened_optimize_platform
    _replace_bound_references(original, hardened_optimize_platform)


def _install_explicit_version_constraint() -> None:
    """Preserve the canonical resolver's non-binding natural-language version hint.

    The optimizer API may still receive an explicit ``version_constraint`` from a
    genuinely target-bound caller.  A version merely mentioned in a new-build prompt,
    however, is not an executable-provider guarantee and must not gate candidate
    discovery.  ``platform_resolver.resolve_platform`` already implements that policy
    and also preserves an existing project's target unless migration was requested.
    """

    return


def _replace_bound_references(original: Any, replacement: Any) -> None:
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("minecraft_mod_ai") or module is None:
            continue
        for attribute, value in tuple(vars(module).items()):
            if value is original:
                setattr(module, attribute, replacement)


def install_pipeline_hardening() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_machine_only_pack_metadata()
    _install_two_stage_platform_optimizer()
    _install_explicit_version_constraint()
    _INSTALLED = True
