from __future__ import annotations

"""Late-bootstrap bridge for supported hot paths that lost their source binding.

Broad cross-module research monkeypatch composition is retired.  This bridge restores
only still-supported behavior whose implementation remains present but became detached
during the compatibility cleanup.  Each repair is narrow, idempotent, and can disappear
once the owning source module absorbs it directly.
"""

import json
import os
import sys
from functools import wraps
from pathlib import Path
from typing import Any


def _restore_validation_fingerprints() -> None:
    from . import validation_execution_contract
    from .research_validation_fingerprint_performance import harden

    harden(validation_execution_contract)


def _restore_managed_research_capacity() -> None:
    from . import central_intelligence_amplifier as central
    from .llama_vram_parallel_policy import validated_active_parallelism

    current = central._research_domain_worker_count
    if getattr(current, "_mmm_receipt_capacity_bridge_v1", False):
        return

    @wraps(current)
    def research_domain_worker_count(router: Any, width: int) -> int:
        requested = min(max(1, int(width)), central._worker_count())
        if not os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "").strip():
            return current(router, width)

        validated = validated_active_parallelism()
        if validated <= 1:
            return 1
        try:
            config = router.registry.role(router.profile, "planner")
        except Exception:
            return 1
        if not bool(getattr(config, "exclusive_gpu", False)):
            return 1
        if str(getattr(config, "provider", "")) != "local":
            return 1
        if str(getattr(config, "adapter", "")) not in {"llama_cpp", "vllm"}:
            return 1
        return max(1, min(requested, validated))

    research_domain_worker_count._mmm_receipt_capacity_bridge_v1 = True
    research_domain_worker_count.__wrapped__ = current
    central._research_domain_worker_count = research_domain_worker_count


def _restore_complete_plan_collection_pages() -> None:
    from . import proposal_store
    from .spec import SpecValidationError

    current = proposal_store.read_sharded_complete_proposal_section
    if getattr(current, "_mmm_collection_page_bridge_v1", False):
        read_section = current
    else:
        @wraps(current)
        def read_section(
            index_path,
            section,
            *,
            cursor="",
            limit=100,
            max_bytes=proposal_store.DEFAULT_PAGE_SIZE_BYTES,
            cursor_key=None,
        ):
            selected = section.strip() if isinstance(section, str) else ""
            if selected not in {"modules", "assets", "acceptance_tests"}:
                return current(
                    index_path,
                    section,
                    cursor=cursor,
                    limit=limit,
                    max_bytes=max_bytes,
                    cursor_key=cursor_key,
                )
            if type(limit) is not int or not 1 <= limit <= proposal_store.MAX_PAGE_ITEMS:
                raise SpecValidationError(
                    f"limit must be between 1 and {proposal_store.MAX_PAGE_ITEMS}."
                )
            if (
                type(max_bytes) is not int
                or not proposal_store.MIN_PAGE_SIZE_BYTES
                <= max_bytes
                <= proposal_store.MAX_PAGE_SIZE_BYTES
            ):
                raise SpecValidationError(
                    "max_bytes must be between "
                    f"{proposal_store.MIN_PAGE_SIZE_BYTES} and "
                    f"{proposal_store.MAX_PAGE_SIZE_BYTES}."
                )
            if cursor_key is not None and (
                not isinstance(cursor_key, bytes) or len(cursor_key) < 16
            ):
                raise SpecValidationError("cursor_key must contain at least 16 bytes.")

            index = Path(index_path).expanduser().resolve()
            try:
                raw = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SpecValidationError(
                    "Complete proposal index is missing or invalid."
                ) from exc
            required = {
                "schema_version",
                "proposal_hash",
                "metadata",
                "base_proposal",
                "game_design",
                "modules",
                "assets",
                "acceptance_tests",
            }
            if (
                not isinstance(raw, dict)
                or set(raw) != required
                or raw.get("schema_version") != proposal_store.INDEX_SCHEMA
            ):
                raise SpecValidationError("Complete proposal shard index fields are invalid.")

            def read_part(relative):
                return proposal_store._read_file_part(index.parent, relative)

            return proposal_store._read_collection_page(
                raw[selected],
                read_part,
                proposal_hash=str(raw["proposal_hash"]),
                section=selected,
                cursor=cursor,
                limit=limit,
                max_bytes=max_bytes,
                cursor_key=cursor_key,
            )

        read_section._mmm_collection_page_bridge_v1 = True
        read_section.__wrapped__ = current
        proposal_store.read_sharded_complete_proposal_section = read_section

    mcp_tools = sys.modules.get(f"{__package__}.mcp_tools")
    if mcp_tools is not None:
        mcp_tools.read_sharded_complete_proposal_section = read_section


def _restore_research_code_context_contracts() -> None:
    from . import research_code_context as research

    cls = research.ResearchCodeContext

    current_entry_points = cls._entry_points
    if not getattr(current_entry_points, "_mmm_semantic_entry_filter_v1", False):
        @wraps(current_entry_points)
        def entry_points(self, query: str):
            candidates = current_entry_points(self, query)
            if not candidates:
                return candidates
            query_tokens = research._tokens(query)
            relevant = [
                symbol
                for symbol in candidates
                if research._overlap(
                    query_tokens,
                    research._tokens(symbol.name)
                    | research._tokens(symbol.signature)
                    | research._tokens(symbol.path),
                )
                > 0.0
            ]
            return relevant or candidates

        entry_points._mmm_semantic_entry_filter_v1 = True
        entry_points.__wrapped__ = current_entry_points
        cls._entry_points = entry_points

    current_expand = cls._expand_partial_graph
    if not getattr(current_expand, "_mmm_two_hop_graph_v1", False):
        @wraps(current_expand)
        def expand_partial_graph(self, entries, *, query=""):
            return [
                (symbol, hop)
                for symbol, hop in current_expand(self, entries, query=query)
                if hop <= 2
            ]

        expand_partial_graph._mmm_two_hop_graph_v1 = True
        expand_partial_graph.__wrapped__ = current_expand
        cls._expand_partial_graph = expand_partial_graph

    current_evolve = cls.evolve_from_generation
    if not getattr(current_evolve, "_mmm_generation_fixed_point_v1", False):
        @wraps(current_evolve)
        def evolve_from_generation(self, text: str):
            digest = research._sha(text)
            seen = getattr(self, "_mmm_generation_evolution_seen", None)
            if not isinstance(seen, set):
                seen = set()
                self._mmm_generation_evolution_seen = seen
            if digest in seen:
                violations = self.monitor.validate_model_output(text)
                return (self.bundle(), violations) if violations else (None, ())
            result = current_evolve(self, text)
            seen.add(digest)
            return result

        evolve_from_generation._mmm_generation_fixed_point_v1 = True
        evolve_from_generation.__wrapped__ = current_evolve
        cls.evolve_from_generation = evolve_from_generation


def install() -> None:
    _restore_validation_fingerprints()
    _restore_managed_research_capacity()
    _restore_complete_plan_collection_pages()
    _restore_research_code_context_contracts()


__all__ = ["install"]
