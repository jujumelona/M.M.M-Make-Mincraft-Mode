from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from functools import wraps
from typing import Any, Iterable

from .platform_catalog import adapter_for_target, supported_minecraft_versions
from .platform_resolver import resolve_platform, retarget_proposal


def install(
    *,
    game_design_module: Any,
    complete_planner_module: Any,
    central_research_module: Any,
    retrieval_module: Any,
    technology_module: Any,
) -> None:
    _install_target_rag(retrieval_module, complete_planner_module)
    _install_dynamic_technology_target(technology_module)
    _install_game_design_target(game_design_module)
    _install_complete_planner_target(
        complete_planner_module,
        central_research_module,
        retrieval_module,
        technology_module,
    )


def _install_game_design_target(module: Any) -> None:
    original_system = module._system_prompt
    if not getattr(original_system, "_mmm_dynamic_target_prompt", False):
        @wraps(original_system)
        def system_prompt() -> str:
            text = original_system()
            text = text.replace(
                "GameDesignPlanner for a Minecraft Java 1.20.1 Fabric production system.",
                "GameDesignPlanner for a Minecraft Java mod production system.",
            )
            return text + (
                "\n\nPlatform target rule: do not choose or assume a Minecraft version, loader, "
                "mappings, Java version, Fabric API, Loom, or Gradle coordinate. The host "
                "resolves one reviewed platform adapter after this design from user constraints, "
                "existing-project metadata, requested capabilities, and compatibility evidence."
            )
        system_prompt._mmm_dynamic_target_prompt = True
        module._system_prompt = system_prompt

    original_sharded = module._sharded_design_system_prompt
    if not getattr(original_sharded, "_mmm_dynamic_target_prompt", False):
        @wraps(original_sharded)
        def sharded_prompt() -> str:
            text = original_sharded()
            text = text.replace(
                "request for a Minecraft Java 1.20.1 Fabric mod.",
                "request for a Minecraft Java mod.",
            )
            return text + (
                "\nThe host, not this page model, selects the exact Minecraft/loader/toolchain "
                "target after all authoritative request pages are merged."
            )
        sharded_prompt._mmm_dynamic_target_prompt = True
        module._sharded_design_system_prompt = sharded_prompt

    original_plan = module.GameDesignPlanner.plan
    if getattr(original_plan, "_mmm_dynamic_platform_resolution", False):
        return

    @wraps(original_plan)
    def plan_with_target(self: Any, prompt: str, *, media_paths=()):
        design, proposal = original_plan(self, prompt, media_paths=media_paths)
        router = self.router
        existing_version = getattr(router, "_mmm_existing_minecraft_version", None)
        existing_loader = getattr(router, "_mmm_existing_loader", None)
        requested_version = getattr(router, "_mmm_requested_minecraft_version", None)
        requested_loader = getattr(router, "_mmm_requested_loader", None)
        effective_prompt = prompt
        if requested_version and requested_version not in prompt:
            effective_prompt += f"\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]"
        if requested_loader and str(requested_loader).casefold() not in prompt.casefold():
            effective_prompt += f"\n[HOST_LOADER_CONSTRAINT {requested_loader}]"
        selection = resolve_platform(
            effective_prompt,
            design=design,
            existing_version=existing_version,
            existing_loader=existing_loader,
        )
        proposal = retarget_proposal(proposal, selection)
        design = {**design, "_platform_selection": selection.to_dict()}
        return design, proposal

    plan_with_target._mmm_dynamic_platform_resolution = True
    module.GameDesignPlanner.plan = plan_with_target


def _install_dynamic_technology_target(module: Any) -> None:
    target_cls = module.TechnologyTarget
    if getattr(target_cls.validate, "_mmm_dynamic_platform_target", False):
        return

    def validate(self: Any) -> None:
        try:
            adapter = adapter_for_target(self.minecraft_version, self.loader)
        except ValueError as exc:
            raise module.SpecValidationError(str(exc)) from exc
        expected = {
            "edition": adapter.edition,
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
            "java_version": adapter.java_version,
            "fabric_loader": adapter.fabric_loader,
            "fabric_api": adapter.fabric_api,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise module.SpecValidationError(
                    f"Technology target is mixed at {field}: expected {value!r}, "
                    f"got {getattr(self, field)!r}."
                )

    validate._mmm_dynamic_platform_target = True
    target_cls.validate = validate

    original_normalize = module.normalize_technology_target
    @wraps(original_normalize)
    def normalize(value: Any):
        if isinstance(value, module.PlatformLock):
            return target_cls(
                edition=value.edition,
                minecraft_version=value.minecraft_version,
                loader=value.loader,
                mappings=value.yarn_mappings,
                java_version=value.java_version,
                fabric_loader=value.fabric_loader,
                fabric_api=value.fabric_api,
            )
        return original_normalize(value)
    normalize._mmm_dynamic_platform_target = True
    module.normalize_technology_target = normalize


def _install_complete_planner_target(
    module: Any,
    central: Any,
    retrieval: Any,
    technology: Any,
) -> None:
    def target_from_design(game_design: dict[str, Any]):
        selection = game_design.get("_platform_selection")
        if not isinstance(selection, dict):
            raise module.SpecValidationError("Planning target selection is missing.")
        target = selection.get("target")
        if not isinstance(target, dict):
            raise module.SpecValidationError("Planning target payload is missing.")
        try:
            return adapter_for_target(
                str(target.get("minecraft_version", "")),
                str(target.get("loader", "fabric")),
            )
        except ValueError as exc:
            raise module.SpecValidationError(str(exc)) from exc

    def retrieve_implementation_evidence(
        prompt: str,
        game_design: dict[str, Any],
        research_brief: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        brief = research_brief or central.normalize_research_brief(prompt, game_design)
        adapter = target_from_design(game_design)
        return _retrieve_domain_evidence_target(
            central,
            retrieval,
            brief,
            adapter=adapter,
        )
    retrieve_implementation_evidence._mmm_dynamic_platform_target = True
    module._retrieve_implementation_evidence = retrieve_implementation_evidence

    # Wrap collection calls already imported into complete_planner so the exact target
    # reaches technology and ecosystem research without changing the public APIs.
    original_tech = module.collect_technology_radar
    if not getattr(original_tech, "_mmm_dynamic_platform_target", False):
        @wraps(original_tech)
        def collect_tech(prompt, research_brief=None, **kwargs):
            target = kwargs.pop("target", None)
            if target is None and isinstance(research_brief, dict):
                target = research_brief.get("_mmm_platform_target")
            return original_tech(prompt, research_brief, target=target, **kwargs)
        collect_tech._mmm_dynamic_platform_target = True
        module.collect_technology_radar = collect_tech

    original_plan = module.CompleteGameDesignPlanner._plan_in_session
    if getattr(original_plan, "_mmm_dynamic_platform_target", False):
        return

    @wraps(original_plan)
    def plan_in_session(self: Any, prompt: str, *, media_paths=(), existing_input_sha256=""):
        # GameDesignPlanner wrapper establishes the target before this method's research
        # phase. We expose the target to imported coordinator functions through a
        # short-lived router attribute so nested calls cannot invent a second target.
        router = self.router
        setattr(router, "_mmm_platform_selection_active", True)
        try:
            result = original_plan(
                self,
                prompt,
                media_paths=media_paths,
                existing_input_sha256=existing_input_sha256,
            )
        finally:
            setattr(router, "_mmm_platform_selection_active", False)

        platform = result.base_proposal.spec.platform
        adapter = adapter_for_target(platform.minecraft_version, platform.loader)
        unsupported = sorted(
            {
                item.kind for item in result.modules
                if item.kind not in adapter.deterministic_module_kinds
            }
        )
        if unsupported:
            raise module.SpecValidationError(
                f"Selected target {adapter.minecraft_version}/{adapter.loader} does not "
                f"have reviewed generation adapters for final module kinds: {unsupported}."
            )
        return result

    plan_in_session._mmm_dynamic_platform_target = True
    module.CompleteGameDesignPlanner._plan_in_session = plan_in_session

    # The original method itself calls collect_technology_radar without a target. Patch
    # the imported function with a target lookup from the design-bearing research brief.
    # normalize_research_brief output receives this host-only key through the wrapper
    # below; it is ignored by domain parsing and removed from public hashing inputs.
    original_normalize = module.normalize_research_brief
    if not getattr(original_normalize, "_mmm_dynamic_platform_target", False):
        @wraps(original_normalize)
        def normalize_brief(prompt: str, game_design: dict[str, Any], candidate=None):
            brief = original_normalize(prompt, game_design, candidate)
            selection = game_design.get("_platform_selection")
            if isinstance(selection, dict) and isinstance(selection.get("target"), dict):
                brief = {**brief, "_mmm_platform_target": dict(selection["target"])}
            return brief
        normalize_brief._mmm_dynamic_platform_target = True
        module.normalize_research_brief = normalize_brief


def _retrieve_domain_evidence_target(
    central: Any,
    retrieval: Any,
    research_brief: dict[str, Any],
    *,
    adapter: Any,
) -> dict[str, Any]:
    domains = research_brief.get("domains")
    if not isinstance(domains, list) or not domains:
        raise central.SpecValidationError("Central research brief has no domains.")
    results: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for raw_domain in domains:
        domain = central._research_domain(raw_domain)
        if "official_docs" not in domain.providers:
            results.append({
                "domain_id": domain.domain_id,
                "strategy": "routed_to_other_providers",
                "queries": [],
            })
            continue
        query_results: list[dict[str, Any]] = []
        has_hits = False
        for query in domain.queries:
            primary = _target_retrieve(
                retrieval,
                query,
                adapter=adapter,
                limit=8,
            )
            corrections: list[dict[str, Any]] = []
            for correction_query in primary.correction_queries:
                correction = _target_retrieve(
                    retrieval,
                    correction_query,
                    adapter=adapter,
                    limit=4,
                )
                corrections.append(correction.to_dict())
                has_hits = has_hits or bool(correction.hits)
            has_hits = has_hits or bool(primary.hits)
            query_results.append({
                "query_sha256": central._sha256(query),
                "strategy": "single" if not primary.correction_required else "corrective_multi_hop",
                "primary": primary.to_dict(),
                "corrections": corrections,
            })
        if not has_hits:
            unresolved.append(domain.domain_id)
        results.append({
            "domain_id": domain.domain_id,
            "strategy": "adaptive_per_query",
            "queries": query_results,
        })
    payload = {
        "schema_version": "mmm/central-evidence-graph-v1",
        "brief_sha256": research_brief.get("brief_sha256", ""),
        "target": {
            "minecraft_version": adapter.minecraft_version,
            "loader": adapter.loader,
            "mappings": adapter.yarn_mappings,
        },
        "domains": results,
        "unresolved_official_domains": unresolved,
        "authorization": "none",
        "retrieval_is_authority": False,
    }
    payload["evidence_sha256"] = central._sha256(central.canonical_json(payload))
    return payload


def _install_target_rag(retrieval: Any, complete_planner: Any) -> None:
    versions = frozenset(supported_minecraft_versions(loader="fabric"))
    retrieval.SUPPORTED_VERSIONS = versions

    existing_ids = {item.document_id for item in retrieval.BUILTIN_CORPUS}
    additions = []
    if "fabric-yarn-1211" not in existing_ids:
        additions.extend(_1211_documents(retrieval))
    merged = tuple(retrieval.BUILTIN_CORPUS) + tuple(additions)
    retrieval.BUILTIN_CORPUS = merged
    complete_planner.BUILTIN_CORPUS = merged
    complete_planner._OFFICIAL_CORPUS_BY_ID = {
        document.document_id: document for document in merged
    }


def _target_retrieve(retrieval: Any, query: str, *, adapter: Any, limit: int):
    documents = retrieval.BUILTIN_CORPUS
    with retrieval.OfficialCorpusIndex(documents=documents) as index:
        if adapter.minecraft_version == "1.20.1":
            return index.retrieve(
                query,
                minecraft_version=adapter.minecraft_version,
                loader=adapter.loader,
                mappings=adapter.yarn_mappings,
                limit=limit,
            )
        # The legacy index has one obsolete hard-coded mappings guard. Pass its guard
        # value only inside the index, then replace all receipt semantics with the
        # selected adapter. Eligible documents and ranking still use 1.21.1.
        receipt = index.retrieve(
            query,
            minecraft_version=adapter.minecraft_version,
            loader=adapter.loader,
            mappings="yarn-1.20.1+build.1",
            limit=limit,
        )
        corrections = tuple(
            value.replace("yarn-1.20.1+build.1", adapter.yarn_mappings)
            for value in receipt.correction_queries
        )
        query_hash = "sha256:" + hashlib.sha256(
            json.dumps(
                {
                    "query": receipt.query,
                    "canonical": receipt.canonical_query,
                    "family": receipt.query_family,
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "mappings": adapter.yarn_mappings,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return replace(
            receipt,
            mappings=adapter.yarn_mappings,
            query_hash=query_hash,
            correction_queries=corrections,
        )


def _1211_documents(retrieval: Any) -> tuple[Any, ...]:
    D = retrieval.CorpusDocument
    common = dict(
        authority="Fabric official documentation or artifact repository",
        trust_tier="official_primary",
        license_id="CC-BY-NC-SA-4.0-or-project-license",
        verified_on="2026-08-12",
        minecraft_versions=("1.21.1",),
        loader="fabric",
        mappings="yarn-1.21.1+build.3",
    )
    return (
        D(
            document_id="fabric-yarn-1211",
            title="Yarn 1.21.1+build.3 Javadoc",
            url="https://maven.fabricmc.net/docs/yarn-1.21.1%2Bbuild.3/",
            revision="minecraft-1.21.1/yarn-1.21.1+build.3",
            families=("source", "content", "profile"),
            topics=("identifier", "item", "block", "registry", "mapping", "javadoc"),
            content=(
                "Exact named Minecraft 1.21.1 API surface for Yarn mappings build 3. "
                "Use Identifier.of and the 1.21.1 named signatures instead of copying 1.20.1 source."
            ),
            related_ids=("fabric-api-1211",),
            **common,
        ),
        D(
            document_id="fabric-api-1211",
            title="Fabric API 0.116.15+1.21.1 artifacts",
            url="https://maven.fabricmc.net/net/fabricmc/fabric-api/fabric-api/0.116.15%2B1.21.1/",
            revision="minecraft-1.21.1/fabric-api-0.116.15",
            families=("profile", "source", "build", "test"),
            topics=("fabric api", "dependency", "artifact", "gametest", "version"),
            content=(
                "Pinned Fabric API artifact lane for Minecraft 1.21.1. Exact target coordinates "
                "must remain bound to generated metadata, source validation and GameTest execution."
            ),
            related_ids=("fabric-yarn-1211",),
            **common,
        ),
        D(
            document_id="java-21-runtime",
            title="Java Platform Standard Edition 21 documentation",
            url="https://docs.oracle.com/en/java/javase/21/",
            authority="Oracle Java SE documentation",
            trust_tier="official_primary",
            license_id="Oracle-documentation-license",
            revision="java-se-21",
            verified_on="2026-08-12",
            minecraft_versions=("1.21.1",),
            loader="agnostic",
            mappings="agnostic",
            families=("profile", "build", "runtime"),
            topics=("java", "jdk", "runtime", "version"),
            content="Official Java SE 21 documentation for the reviewed Minecraft 1.21.1 adapter.",
            related_ids=("fabric-yarn-1211",),
        ),
    )
