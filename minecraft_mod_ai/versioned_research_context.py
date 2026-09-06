from __future__ import annotations

"""ResearchCodeContext extension that adds exact-version external exemplars."""

import os

from typing import Any

from .research_code_context import (
    Evidence,
    PlanStep,
    QualityVector,
    ResearchCodeContext,
    _code_plan,
    _sha,
)
from .versioned_reference_catalog import VersionedReferenceCatalog


class VersionedResearchCodeContext(ResearchCodeContext):
    """Preserve local iterative retrieval and add a bounded external reference lane."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._versioned_reference_catalog = VersionedReferenceCatalog(
            minecraft_version=self.minecraft_version,
            loader=self.loader,
            mappings=self.mappings,
        )
        raw_budget = os.environ.get("MMM_CODE_EXTERNAL_REFERENCE_QUERIES", "12").strip()
        try:
            configured_budget = int(raw_budget)
        except ValueError:
            configured_budget = 12
        self._versioned_reference_query_budget = max(2, min(32, configured_budget))
        self._versioned_reference_queries: set[str] = set()

    def _retrieve_repo_examples(
        self,
        query: str,
        *,
        plan_step: PlanStep | None,
    ) -> list[Evidence]:
        # The generated workspace stays first.  External repositories supplement rather
        # than replace the current-project call graph, RAG and quality-aware retrieval.
        local = super()._retrieve_repo_examples(query, plan_step=plan_step)
        capability = plan_step.capability if plan_step is not None else ""
        query_key = _sha(
            {
                "query": " ".join(query.split()).casefold(),
                "capability": capability.casefold(),
            }
        )
        if (
            query_key not in self._versioned_reference_queries
            and len(self._versioned_reference_queries)
            >= self._versioned_reference_query_budget
        ):
            return local
        self._versioned_reference_queries.add(query_key)

        external_budget = min(12 * 1024, max(3072, self.byte_budget // 2))
        try:
            excerpts = self._versioned_reference_catalog.retrieve(
                query,
                capability=capability,
                max_examples=3,
                byte_budget=external_budget,
            )
        except Exception as exc:
            # External exemplars are evidence, never an availability dependency.  A
            # network/provider failure leaves the existing local+official retrieval path
            # intact and records why the optional lane contributed nothing.
            self.rounds.append(
                {
                    "trigger": "versioned_reference_unavailable",
                    "query_sha256": _sha(query),
                    "target": {
                        "minecraft_version": self.minecraft_version,
                        "loader": self.loader,
                        "mappings": self.mappings,
                    },
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                }
            )
            return local

        external: list[Evidence] = []
        for excerpt in excerpts:
            quality = QualityVector(
                correctness=0.94,
                efficiency=0.86,
                security=0.88,
                maintainability=0.92,
                complexity_fit=0.90,
                readability=0.88,
                stepwise_clarity=0.86,
            )
            metrics = {
                "retrieval_score": excerpt.score,
                "external_reference": 1.0,
                "target_exact": 1.0,
                "immutable_commit": 1.0,
                "license_admitted": 1.0,
                "fabric_api_exact": 1.0
                if excerpt.compatibility.get("fabric_api_exact")
                else 0.0,
                "loader_version_exact": 1.0
                if excerpt.compatibility.get("loader_version_exact")
                else 0.0,
                "mappings_exact": 1.0
                if excerpt.compatibility.get("mappings_exact")
                else 0.0,
            }
            evidence = Evidence(
                evidence_id=excerpt.evidence_id(),
                source_type="repository_external_versioned",
                path=(
                    f"github:{excerpt.repository}@{excerpt.commit_sha}:"
                    f"{excerpt.path}"
                ),
                text=(
                    "EXTERNAL_REFERENCE_METADATA "
                    f"repository={excerpt.repository} "
                    f"commit={excerpt.commit_sha} "
                    f"ref={excerpt.ref_name} "
                    f"license={excerpt.license_id} "
                    f"minecraft={excerpt.minecraft_version} "
                    f"loader={excerpt.loader} "
                    f"compatibility_sha256={_sha(dict(excerpt.compatibility))}\n"
                    + excerpt.text
                ),
                sha256=excerpt.sha256,
                start_line=excerpt.start_line,
                end_line=excerpt.end_line,
                symbols=excerpt.symbols,
                metrics=metrics,
                quality=quality,
                bestfit_score=excerpt.score,
                algorithmic_plan=_code_plan(excerpt.text),
            )
            if plan_step is not None:
                evidence.plan_steps.add(plan_step.step_id)
            self._merge_evidence(evidence)
            self._evolve_knowledge(evidence.text)
            external.append(evidence)

        if external:
            self.rounds.append(
                {
                    "trigger": "versioned_external_reference",
                    "query_sha256": _sha(query),
                    "target": {
                        "minecraft_version": self.minecraft_version,
                        "loader": self.loader,
                        "mappings": self.mappings,
                    },
                    "example_count": len(external),
                    "repositories": sorted(
                        {
                            item.path.split("@", 1)[0].removeprefix("github:")
                            for item in external
                        }
                    ),
                    "evidence_sha256": _sha(
                        [item.evidence_id for item in external]
                    ),
                }
            )
        return [*local, *external]


__all__ = ["VersionedResearchCodeContext"]
