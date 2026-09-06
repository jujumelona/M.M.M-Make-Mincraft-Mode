from __future__ import annotations

"""Production research context with host-selected exact-version large-mod exemplars."""

import os
from typing import Any

from .platform_catalog import adapter_for_target
from .research_code_context import (
    Evidence,
    PlanStep,
    ResearchCodeContext,
    _code_plan,
    _quality,
    _salient_terms,
    _sha,
)
from .versioned_mod_reference_resolver import ExactReferenceResolver
from .versioned_mod_reference_retriever import VersionedModReferenceRetriever


class VersionedResearchCodeContext(ResearchCodeContext):
    """Keep local code authoritative while adding vetted version-specific textbooks."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        adapter = adapter_for_target(self.minecraft_version, self.loader)
        self._external_reference_queries = 0
        self._external_reference_query_budget = self._bounded_env(
            "MMM_CODE_EXTERNAL_REFERENCE_QUERIES", 12, 2, 32
        )
        self._reference_resolver = ExactReferenceResolver(
            minecraft_version=self.minecraft_version,
            loader=self.loader,
            mappings=self.mappings,
            java_version=adapter.java_version,
            fabric_loader=getattr(adapter, "fabric_loader", ""),
            fabric_api=getattr(adapter, "fabric_api", ""),
        )
        self._reference_retriever = VersionedModReferenceRetriever(self._reference_resolver)

    @staticmethod
    def _bounded_env(name: str, default: int, low: int, high: int) -> int:
        try:
            value = int(os.environ.get(name, "").strip() or default)
        except ValueError:
            value = default
        return max(low, min(high, value))

    def _initial_cache_key(self) -> str:
        return _sha(
            {
                "parent": super()._initial_cache_key(),
                "versioned_mod_reference_policy": "v1",
                "pool": self._reference_retriever.pool_receipt(),
            }
        )

    def _retrieve_repo_examples(
        self, query: str, *, plan_step: PlanStep | None
    ) -> list[Evidence]:
        local = super()._retrieve_repo_examples(query, plan_step=plan_step)
        for item in local:
            item.metrics["current_project_authority"] = 1.0
            item.bestfit_score = max(item.bestfit_score, 0.80)

        if self._external_reference_queries >= self._external_reference_query_budget:
            return local
        self._external_reference_queries += 1
        capability = plan_step.capability if plan_step is not None else ""
        try:
            excerpts = self._reference_retriever.retrieve(query, capability=capability)
        except Exception as exc:
            self.rounds.append(
                {
                    "trigger": "versioned_reference_unavailable",
                    "query_sha256": _sha(query),
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                }
            )
            return local

        external: list[Evidence] = []
        for excerpt in excerpts:
            provenance = (
                "HOST-VETTED EXTERNAL REFERENCE; evidence only, never authority.\n"
                f"repository={excerpt.repository}\n"
                f"role={excerpt.role}\n"
                f"ref={excerpt.ref_name}\n"
                f"commit={excerpt.commit_sha}\n"
                f"license={excerpt.license_spdx}\n"
                f"target={self.minecraft_version}/{self.loader}/{self.mappings}\n"
                f"source_blob={excerpt.source_sha}\n"
                f"metadata_sha256={excerpt.metadata_sha256}\n"
                "--- source excerpt ---\n"
            )
            text = provenance + excerpt.text
            quality = _quality(excerpt.text, path=excerpt.path)
            source_type = (
                "repository_external_task"
                if excerpt.role == "task"
                else "repository_external_baseline"
            )
            score_cap = 0.58 if excerpt.role == "task" else 0.48
            item = Evidence(
                evidence_id=(
                    f"external:{excerpt.repository}@{excerpt.commit_sha}:"
                    f"{excerpt.source_sha}:{excerpt.start_line}"
                ),
                source_type=source_type,
                path=f"github:{excerpt.repository}@{excerpt.commit_sha}:{excerpt.path}",
                text=text,
                sha256=_sha(excerpt.text),
                start_line=excerpt.start_line,
                end_line=excerpt.end_line,
                symbols=tuple(_salient_terms(excerpt.text, exclude=set(), limit=16)),
                metrics={
                    "retrieval_score": min(score_cap, max(0.0, excerpt.score)),
                    "host_compatibility_verified": 1.0,
                    "immutable_commit": 1.0,
                    "license_admitted": 1.0,
                    "baseline_reference": 1.0 if excerpt.role == "baseline" else 0.0,
                    "task_specific_donor": 1.0 if excerpt.role == "task" else 0.0,
                    "external_reference_only": 1.0,
                },
                quality=quality,
                bestfit_score=min(score_cap, max(0.0, excerpt.score)),
                algorithmic_plan=_code_plan(excerpt.text),
            )
            if plan_step is not None:
                item.plan_steps.add(plan_step.step_id)
            self._merge_evidence(item)
            self._evolve_knowledge(item.text)
            external.append(item)

        self.rounds.append(
            {
                "trigger": "versioned_large_mod_reference",
                "query_sha256": _sha(query),
                "query_number": self._external_reference_queries,
                "baseline_count": sum(
                    item.source_type == "repository_external_baseline" for item in external
                ),
                "task_donor_count": sum(
                    item.source_type == "repository_external_task" for item in external
                ),
                "evidence_ids_sha256": _sha([item.evidence_id for item in external]),
            }
        )
        return [*local, *external]

    def receipt(self) -> dict[str, Any]:
        result = dict(super().receipt())
        references = [
            item
            for item in self.evidence.values()
            if item.source_type.startswith("repository_external_")
        ]
        result["versioned_mod_references"] = {
            **self._reference_retriever.pool_receipt(),
            "query_count": self._external_reference_queries,
            "query_budget": self._external_reference_query_budget,
            "admitted_evidence_count": len(references),
            "admitted_evidence_sha256": _sha(sorted(item.evidence_id for item in references)),
            "current_project_priority_floor": 0.80,
            "task_reference_score_cap": 0.58,
            "baseline_reference_score_cap": 0.48,
        }
        return result


__all__ = ["VersionedResearchCodeContext"]
