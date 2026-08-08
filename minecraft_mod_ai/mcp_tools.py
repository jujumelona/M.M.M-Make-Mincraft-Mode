from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .broker import LocalPolicyBroker, ToolAction, approved_request
from .complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import CompleteProposal
from .conversation import merge_design_brief
from .ecosystem_discovery import EcosystemDiscoveryClient
from .game_design import GameDesignPlanner
from .importer import inspect_existing_project_archive
from .knowledge import AuthoritativeEvidenceRetriever
from .model_router import ModelRouter
from .plan_render import render_complete_plan
from .proposal_store import (
    load_sharded_complete_proposal,
    read_sharded_complete_proposal_section,
    write_sharded_complete_proposal,
)
from .production_contract import quality_contract_summary, quality_unresolved
from .repair_engine import RepairEngine
from .runner import GradleRunner
from .scalable_pipeline import ScalableMinecraftModPipeline
from .scalable_validator import ScalableProjectValidator
from .scale_policy import ScalePolicy
from .source_patch import TransactionalSourcePatcher
from .spec import (
    Proposal,
    ProposalStatus,
    SpecValidationError,
    canonical_json,
)
from .technology_radar import (
    _assess_technology_candidate_with_receipt_key as assess_technology_candidate,
    _build_signed_official_target_evidence,
    build_technology_radar as create_technology_radar,
)
from .validator import validate_jar

_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PROPOSAL_REF = re.compile(
    r"^plan_([0-9a-f]{64})_([0-9a-f]{64})$"
)
_RUN_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


class MMMToolService:
    """Concrete tool service for the stdio MCP server and compatibility gateway."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = "mmm-output",
        profile: str = "t4_local",
        router_factory: Callable[[], ModelRouter] | None = None,
        discovery_client_factory: Callable[[], EcosystemDiscoveryClient]
        | None = None,
        policy: ScalePolicy | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.router_factory = router_factory or (
            lambda: ModelRouter(profile=profile)
        )
        self.discovery_client_factory = discovery_client_factory or (
            EcosystemDiscoveryClient
        )
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.broker = LocalPolicyBroker()
        self._technology_receipt_key = secrets.token_bytes(32)

    def discover_ecosystem_resources(
        self,
        provider: str,
        query: str,
        cursor: str = "",
        limit: int = 20,
        target_profile: str = "minecraft_mod",
    ) -> dict[str, Any]:
        """Search one read-only, evidence-only ecosystem page."""

        return self.discovery_client_factory().search(
            provider,
            query,
            cursor=cursor,
            limit=limit,
            target_profile=target_profile,
        )

    def inspect_modrinth_project(self, project_id: str) -> dict[str, Any]:
        """Inspect exact Fabric 1.20.1 versions without downloading a JAR."""

        return self.discovery_client_factory().inspect_modrinth_project(
            project_id
        )

    def inspect_github_repository(self, full_name: str) -> dict[str, Any]:
        """Pin a public commit and license receipt without cloning the source."""

        return self.discovery_client_factory().inspect_github_repository(
            full_name
        )

    def inspect_huggingface_model(self, repo_id: str) -> dict[str, Any]:
        """Pin and inspect public model metadata without downloading weights."""

        return self.discovery_client_factory().inspect_huggingface_model(
            repo_id
        )

    @staticmethod
    def build_technology_radar(
        prompt: str,
        research_brief: dict[str, Any] | None = None,
        cursor: str = "",
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Page request-derived AI and speech requirements read-only."""

        return create_technology_radar(
            prompt,
            research_brief,
            cursor=cursor,
            page_size=page_size,
        )

    def assess_technology_compatibility(
        self,
        requirement: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Fail closed on compatibility, license, consent and benchmark gates."""

        assessed_candidate = dict(candidate)
        assessed_candidate["official_target_evidence"] = (
            _build_signed_official_target_evidence(
                requirement,
                receipt_key=self._technology_receipt_key,
            )
        )
        return assess_technology_candidate(
            requirement,
            assessed_candidate,
            receipt_key=self._technology_receipt_key,
        )

    def plan_game(
        self,
        prompt: str,
        media_paths: Sequence[str] = (),
    ) -> dict[str, Any]:
        planner = GameDesignPlanner(self.router_factory())
        design, proposal = planner.plan(
            prompt,
            media_paths=self._scoped_media_paths(media_paths),
        )
        return {
            "schema_version": "mmm/plan-result-v2",
            "profile": self.profile,
            "game_design": design,
            "proposal": proposal.to_dict(),
            "approval_hash": proposal.calculate_hash(),
        }

    def plan_complete_game(
        self,
        prompt: str,
        media_paths: Sequence[str] = (),
        existing_input_sha256: str = "",
    ) -> dict[str, Any]:
        proposal = CompleteGameDesignPlanner(self.router_factory()).plan(
            prompt,
            media_paths=self._scoped_media_paths(media_paths),
            existing_input_sha256=existing_input_sha256,
        )
        proposal_ref = self._store_complete_proposal(proposal)
        return {
            "schema_version": "mmm/complete-plan-result-v3",
            "profile": self.profile,
            "message": render_complete_plan(
                requested_prompt=proposal.requested_prompt,
                game_design=proposal.game_design,
                modules=proposal.modules,
                acceptance_tests=proposal.acceptance_tests,
            ),
            "proposal_ref": proposal_ref,
            "approval_hash": proposal.calculate_hash(),
            "counts": self._complete_proposal_counts(proposal),
            "detail_tool": "read_complete_plan_section",
        }

    def revise_complete_plan(
        self,
        original_prompt: str,
        revision: str,
        media_paths: Sequence[str] = (),
        existing_input_sha256: str = "",
    ) -> dict[str, Any]:
        try:
            merged = merge_design_brief(original_prompt, revision)
        except ValueError as exc:
            raise SpecValidationError("revision must not be empty.") from exc
        return self.plan_complete_game(
            merged,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
        )

    def approve_complete_plan(
        self,
        complete_proposal: dict[str, Any] | None = None,
        approval_hash: str = "",
        proposal_ref: str = "",
    ) -> dict[str, Any]:
        parsed, stored_ref = self._resolve_complete_proposal(
            complete_proposal=complete_proposal,
            proposal_ref=proposal_ref,
        )
        approved = parsed.approve(approval_hash, policy=self.policy)
        return {
            "schema_version": "mmm/complete-plan-approval-v2",
            "status": approved.status.value,
            "proposal_ref": stored_ref,
            "approval_hash": approved.calculate_hash(),
            "counts": self._complete_proposal_counts(approved),
        }

    def execute_complete_project(
        self,
        complete_proposal: dict[str, Any] | None = None,
        approval_hash: str = "",
        run_name: str = "",
        options: dict[str, Any] | None = None,
        existing_input: str | None = None,
        proposal_ref: str = "",
    ) -> dict[str, Any]:
        parsed, _ = self._resolve_complete_proposal(
            complete_proposal=complete_proposal,
            proposal_ref=proposal_ref,
        )
        scoped_options = dict(options or {})
        if scoped_options.get("server_launcher"):
            scoped_options["server_launcher"] = str(
                self._existing_file(str(scoped_options["server_launcher"]))
            )
        if scoped_options.get("screenshot_paths"):
            raw_screenshots = scoped_options["screenshot_paths"]
            if not isinstance(raw_screenshots, (list, tuple)):
                raise SpecValidationError(
                    "screenshot_paths must be a list of workspace files."
                )
            scoped_options["screenshot_paths"] = tuple(
                str(self._existing_file(str(value)))
                for value in raw_screenshots
            )
        parsed_options = CompleteExecutionOptions(**scoped_options)
        scoped_existing = (
            str(self._existing_file(existing_input))
            if existing_input is not None
            else None
        )
        return CompleteProductionOrchestrator(
            workspace_root=self.workspace_root,
            profile=self.profile,
            router_factory=self.router_factory,
            policy=self.policy,
        ).execute(
            parsed,
            approval_hash=approval_hash,
            run_name=run_name,
            options=parsed_options,
            existing_input=scoped_existing,
        ).to_dict()

    def read_complete_plan_section(
        self,
        proposal_ref: str,
        section: str = "overview",
        cursor: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """Read one bounded proposal page without transferring the full plan."""

        index, expected_hash, expected_index_hash = self._proposal_index_for_ref(
            proposal_ref,
            require_existing=True,
        )
        if _sha256(index) != expected_index_hash:
            raise SpecValidationError(
                "Stored proposal index does not match its opaque reference."
            )
        result = read_sharded_complete_proposal_section(
            index,
            section,
            cursor=cursor,
            limit=limit,
            max_bytes=self.policy.mcp_page_bytes,
            cursor_key=self._proposal_cursor_key(
                index,
                create=False,
            ),
        )
        if result.get("proposal_hash") != expected_hash:
            raise SpecValidationError(
                "Stored proposal does not match its opaque reference."
            )
        return {
            **result,
            "proposal_ref": proposal_ref,
        }

    def read_quality_contract(self, proposal_ref: str) -> dict[str, Any]:
        """Read the bounded completion contract for one stored proposal."""

        proposal, stored_ref = self._resolve_complete_proposal(
            complete_proposal=None,
            proposal_ref=proposal_ref,
        )
        contract = proposal.game_design.get("_production_contract")
        if not isinstance(contract, dict):
            raise SpecValidationError(
                "This legacy proposal has no production quality contract."
            )
        return {
            "schema_version": "mmm/quality-contract-summary-v1",
            "proposal_ref": stored_ref,
            "summary": quality_contract_summary(contract),
            "catalog_stats": dict(contract["catalog_stats"]),
            "quality_dimensions": [
                {
                    "dimension_id": item["dimension_id"],
                    "title": item["title"],
                    "activation": item["activation"],
                    "objective": item["objective"],
                    "evidence_route_ref": item["evidence_route_ref"],
                }
                for item in contract["quality_dimension_catalog"]
            ],
            "completion_policy": dict(contract["completion_policy"]),
        }

    def quality_status(self, run_name: str) -> dict[str, Any]:
        """Read and validate a persisted quality-convergence report."""

        if not isinstance(run_name, str) or not _RUN_NAME.fullmatch(run_name):
            raise SpecValidationError(
                "run_name must use lowercase letters, numbers, underscore or hyphen."
            )
        run_root = (self.workspace_root / run_name).resolve()
        try:
            run_root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SpecValidationError("Run path escaped the workspace.") from exc
        target = (run_root / ".minecraft_ai/quality-convergence.json").resolve()
        try:
            target.relative_to(run_root)
        except ValueError as exc:
            raise SpecValidationError("Quality report escaped the run directory.") from exc
        if run_root.is_symlink() or target.is_symlink() or not target.is_file():
            raise FileNotFoundError(
                f"Quality report not found for run: {run_name}"
            )
        if target.stat().st_size > self.policy.mcp_page_bytes * 8:
            raise SpecValidationError("Quality report exceeds the read size policy.")
        try:
            report = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SpecValidationError(f"Invalid quality report: {exc}") from exc
        if not isinstance(report, dict):
            raise SpecValidationError("Quality report must be an object.")
        unresolved = quality_unresolved(report)
        return {
            **report,
            "run_name": run_name,
            "unresolved_dimension_ids": list(unresolved),
        }

    def apply_source_patch(
        self,
        project_root: str,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        root = self._existing_dir(project_root)
        encoded = json.dumps(
            operations,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > self.policy.max_patch_bytes:
            raise SpecValidationError(
                "Patch exceeds MMM_MAX_PATCH_BYTES host resource policy."
            )
        return TransactionalSourcePatcher(root).apply(operations)

    def repair_project(
        self,
        project_root: str,
        run_gametest: bool = True,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        root = self._existing_dir(project_root)
        attempts = (
            self.policy.repair_attempts
            if max_attempts is None
            else max_attempts
        )
        if type(attempts) is not int or attempts < 1:
            raise SpecValidationError(
                "max_attempts must be null or a positive integer."
            )
        return RepairEngine(
            router=self.router_factory(),
            gradle_cache=self.workspace_root / ".cache" / "gradle",
            policy=self.policy,
        ).repair(
            root,
            run_gametest=run_gametest,
            max_attempts=attempts,
        )

    def revise_plan(
        self,
        original_prompt: str,
        revision: str,
        media_paths: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not revision.strip():
            raise SpecValidationError("revision must not be empty.")
        merged = (
            f"{original_prompt.strip()}\n\n"
            f"User revision:\n{revision.strip()}"
        )
        return self.plan_game(merged, media_paths=media_paths)

    def approve_plan(
        self,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        parsed = Proposal.from_dict(proposal)
        approved = parsed.approve(approval_hash)
        approved.validate()
        return {
            "status": approved.status.value,
            "proposal": approved.to_dict(),
            "approval_hash": approved.calculate_hash(),
        }

    def search_project_rag(
        self,
        query: str,
        minecraft_version: str = "1.20.1",
        limit: int = 6,
    ) -> dict[str, Any]:
        if type(limit) is not int or limit < 1:
            raise SpecValidationError("limit must be a positive integer.")
        sources = AuthoritativeEvidenceRetriever().search(
            query,
            minecraft_version=minecraft_version,
            limit=limit,
        )
        return {
            "schema_version": "mmm/rag-result-v1",
            "query": query,
            "minecraft_version": minecraft_version,
            "sources": [source.__dict__ for source in sources],
        }

    def inspect_existing_mod(self, archive_path: str) -> dict[str, Any]:
        archive = self._existing_file(archive_path)
        return inspect_existing_project_archive(archive).to_dict()

    def generate_fabric_project(
        self,
        proposal: dict[str, Any],
        approval_hash: str,
        run_name: str = "mcp-run",
    ) -> dict[str, Any]:
        parsed = Proposal.from_dict(proposal)
        run_root = self._new_child(run_name)
        result = ScalableMinecraftModPipeline(
            policy=self.policy
        ).execute(
            parsed,
            approval_hash=approval_hash,
            output_root=run_root,
            build=False,
            run_gametest=False,
        )
        return result.to_dict()

    def generate_assets(
        self,
        assets: dict[str, str],
        output_dir: str = "assets-generated",
        seed: int = 0,
    ) -> dict[str, Any]:
        if not isinstance(assets, dict) or not assets:
            raise SpecValidationError(
                "assets must be a non-empty id-to-prompt object."
            )
        if type(seed) is not int:
            raise SpecValidationError("seed must be a JSON integer.")
        try:
            encoded = json.dumps(
                assets,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SpecValidationError(
                "assets must contain JSON string prompts."
            ) from exc
        if len(encoded) > self.policy.max_patch_bytes:
            raise SpecValidationError(
                "Asset request exceeds MMM_MAX_PATCH_BYTES host resource policy."
            )

        normalized: list[tuple[str, str]] = []
        for asset_id, prompt in assets.items():
            if not isinstance(asset_id, str) or not _SAFE_ID.fullmatch(asset_id):
                raise SpecValidationError(
                    f"Invalid asset id: {asset_id!r}"
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise SpecValidationError(
                    f"Asset prompt is empty: {asset_id}"
                )
            if len(prompt.encode("utf-8")) > self.policy.max_single_file_bytes:
                raise SpecValidationError(
                    f"Asset prompt exceeds host resource policy: {asset_id}"
                )
            normalized.append((asset_id, prompt.strip()))

        target = self._new_child(output_dir)
        concepts = target / "concepts"
        textures = target / "textures"
        concepts.mkdir(parents=True, exist_ok=True)
        textures.mkdir(parents=True, exist_ok=True)
        router = self.router_factory()
        generated: list[dict[str, str]] = []
        for index, (asset_id, prompt) in enumerate(sorted(normalized)):
            concept = router.generate_image(
                "image_generator",
                prompt=(
                    "Minecraft Java resource-pack source art, centered object, "
                    "clean silhouette, no text, no watermark, square composition. "
                    + prompt
                ),
                output_path=concepts / f"{asset_id}.png",
                width=512,
                height=512,
                seed=seed + index,
            )
            texture = textures / f"{asset_id}.png"
            _minecraft_texture(concept, texture)
            generated.append(
                {
                    "asset_id": asset_id,
                    "concept_png": str(concept),
                    "texture_png": str(texture),
                    "concept_sha256": _sha256(concept),
                    "texture_sha256": _sha256(texture),
                }
            )
        return {
            "schema_version": "mmm/asset-result-v2",
            "output_dir": str(target),
            "generated": generated,
            "asset_count": len(generated),
            "warning": (
                "Generated textures require VisualCritic review before release."
            ),
        }

    def run_static_validation(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        approved = self._approved(proposal, approval_hash)
        root = self._existing_dir(project_root)
        self.broker.authorize(
            approved_request(
                ToolAction.VALIDATE,
                project_root=root,
                workspace_root=self.workspace_root,
                proposal=approved,
            ),
            approved,
        )
        return ScalableProjectValidator(
            policy=self.policy
        ).validate(root, approved.spec).to_dict()

    def run_gradle_build(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        return self._run_gradle(
            project_root,
            proposal,
            approval_hash,
            run_gametest=False,
        )

    def run_gametest(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        return self._run_gradle(
            project_root,
            proposal,
            approval_hash,
            run_gametest=True,
        )

    def inspect_jar(
        self,
        jar_path: str,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        parsed = Proposal.from_dict(proposal)
        jar = self._existing_file(jar_path)
        report = validate_jar(jar, parsed.spec)
        return {
            **report.to_dict(),
            "jar_path": str(jar),
            "jar_sha256": _sha256(jar),
        }

    def package_release(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
        output_zip: str = "releases/mmm-release.zip",
        jar_path: str | None = None,
    ) -> dict[str, Any]:
        approved = self._approved(proposal, approval_hash)
        root = self._existing_dir(project_root)
        self.broker.authorize(
            approved_request(
                ToolAction.PACKAGE,
                project_root=root,
                workspace_root=self.workspace_root,
                proposal=approved,
            ),
            approved,
        )
        source_report = ScalableProjectValidator(
            policy=self.policy
        ).validate(root, approved.spec)
        if not source_report.passed:
            raise RuntimeError(
                "Source validation failed; release package was not created."
            )
        jar: Path | None = None
        jar_report: dict[str, Any] | None = None
        if jar_path is not None:
            jar = self._existing_file(jar_path)
            validated = validate_jar(jar, approved.spec)
            if not validated.passed:
                raise RuntimeError(
                    "JAR validation failed; release package was not created."
                )
            jar_report = validated.to_dict()
        target = self._new_file(output_zip)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Release already exists: {target}")
        with zipfile.ZipFile(
            target,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zipped:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                if any(
                    part in {".gradle", ".cache", "gradle-user-home", "run"}
                    for part in relative.parts
                ):
                    continue
                zipped.write(path, Path("source") / relative)
            if jar is not None:
                zipped.write(jar, Path("binary") / jar.name)
            zipped.writestr(
                "release-manifest.json",
                canonical_json(
                    {
                        "schema_version": "mmm/release-manifest-v2",
                        "proposal_hash": approved.calculate_hash(),
                        "source_validation": source_report.to_dict(),
                        "jar_validation": jar_report,
                        "resource_policy": self.policy.__dict__,
                    }
                ),
            )
        return {
            "status": "PACKAGED",
            "release_zip": str(target),
            "sha256": _sha256(target),
            "includes_verified_jar": jar is not None,
        }

    def _run_gradle(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
        *,
        run_gametest: bool,
    ) -> dict[str, Any]:
        approved = self._approved(proposal, approval_hash)
        root = self._existing_dir(project_root)
        action = (
            ToolAction.GAME_TEST
            if run_gametest
            else ToolAction.GRADLE_BUILD
        )
        self.broker.authorize(
            approved_request(
                action,
                project_root=root,
                workspace_root=self.workspace_root,
                proposal=approved,
            ),
            approved,
        )
        cache = self.workspace_root / ".cache" / "gradle"
        return GradleRunner(cache).build(
            root,
            run_gametest=run_gametest,
        ).to_dict()

    @staticmethod
    def _approved(
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> Proposal:
        parsed = Proposal.from_dict(proposal)
        approved = parsed.approve(approval_hash)
        if approved.status is not ProposalStatus.APPROVED:
            raise SpecValidationError("Proposal approval failed.")
        return approved

    def _store_complete_proposal(
        self,
        proposal: CompleteProposal,
    ) -> str:
        proposal.validate(policy=self.policy)
        digest = proposal.calculate_hash().removeprefix("sha256:")
        expected_hash = f"sha256:{digest}"
        index = self._proposal_index_for_digest(
            digest,
            require_existing=False,
        )
        if index.exists():
            if not index.is_file() or index.is_symlink():
                raise SpecValidationError(
                    "Stored proposal index is unsafe."
                )
            stored = load_sharded_complete_proposal(index)
            if stored.calculate_hash() != expected_hash:
                raise SpecValidationError(
                    "Stored proposal does not match its opaque reference."
                )
            self._proposal_cursor_key(index, create=True)
            return (
                f"plan_{digest}_"
                f"{_sha256(index).removeprefix('sha256:')}"
            )
        index.parent.mkdir(parents=True, exist_ok=True)
        if index.parent.is_symlink():
            raise SpecValidationError(
                "Stored proposal directory may not be a symlink."
            )
        write_sharded_complete_proposal(
            proposal,
            index,
            shard_size=self.policy.java_shard_size,
            policy=self.policy,
        )
        stored = load_sharded_complete_proposal(index)
        if stored.calculate_hash() != expected_hash:
            raise SpecValidationError(
                "Stored proposal verification failed."
            )
        self._proposal_cursor_key(index, create=True)
        return (
            f"plan_{digest}_"
            f"{_sha256(index).removeprefix('sha256:')}"
        )

    def _resolve_complete_proposal(
        self,
        *,
        complete_proposal: dict[str, Any] | None,
        proposal_ref: str,
    ) -> tuple[CompleteProposal, str]:
        has_inline = complete_proposal is not None
        has_ref = bool(proposal_ref)
        if has_inline == has_ref:
            raise SpecValidationError(
                "Supply exactly one of complete_proposal or proposal_ref."
            )
        if has_inline:
            if not isinstance(complete_proposal, dict):
                raise SpecValidationError(
                    "complete_proposal must be an object."
                )
            parsed = CompleteProposal.from_dict(complete_proposal)
            return parsed, self._store_complete_proposal(parsed)
        index, expected_hash, expected_index_hash = self._proposal_index_for_ref(
            proposal_ref,
            require_existing=True,
        )
        if _sha256(index) != expected_index_hash:
            raise SpecValidationError(
                "Stored proposal index does not match its opaque reference."
            )
        parsed = load_sharded_complete_proposal(index)
        if parsed.calculate_hash() != expected_hash:
            raise SpecValidationError(
                "Stored proposal does not match its opaque reference."
            )
        return parsed, proposal_ref

    def _proposal_index_for_ref(
        self,
        proposal_ref: str,
        *,
        require_existing: bool,
    ) -> tuple[Path, str, str]:
        if not isinstance(proposal_ref, str):
            raise SpecValidationError("proposal_ref must be a string.")
        match = _PROPOSAL_REF.fullmatch(proposal_ref)
        if match is None:
            raise SpecValidationError(
                "proposal_ref is not a valid M.M.M plan reference."
            )
        digest = match.group(1)
        index_digest = match.group(2)
        index = self._proposal_index_for_digest(
            digest,
            require_existing=require_existing,
        )
        return (
            index,
            f"sha256:{digest}",
            f"sha256:{index_digest}",
        )

    def _proposal_index_for_digest(
        self,
        digest: str,
        *,
        require_existing: bool,
    ) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SpecValidationError("Proposal digest is invalid.")
        logical_parts = (
            self.workspace_root / ".minecraft_ai",
            self.workspace_root / ".minecraft_ai" / "plans",
            self.workspace_root / ".minecraft_ai" / "plans" / digest,
        )
        for part in logical_parts:
            if part.exists() and part.is_symlink():
                raise SpecValidationError(
                    "Proposal storage directories may not be symlinks."
                )
        index = (logical_parts[-1] / "complete-proposal.json").resolve()
        try:
            index.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SpecValidationError(
                "Proposal reference escaped the configured workspace."
            ) from exc
        if require_existing and (
            not index.is_file() or index.is_symlink()
        ):
            raise SpecValidationError(
                "Stored proposal reference was not found."
            )
        return index

    def _proposal_cursor_key(
        self,
        index: Path,
        *,
        create: bool,
    ) -> bytes:
        key_path = (index.parent / "cursor.key").resolve()
        try:
            key_path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SpecValidationError(
                "Proposal cursor key escaped the configured workspace."
            ) from exc
        if key_path.exists():
            if not key_path.is_file() or key_path.is_symlink():
                raise SpecValidationError(
                    "Proposal cursor key is unsafe."
                )
            value = key_path.read_bytes()
            if len(value) != 32:
                raise SpecValidationError(
                    "Proposal cursor key has an invalid size."
                )
            return value
        if not create:
            raise SpecValidationError(
                "Proposal cursor key is missing."
            )
        value = os.urandom(32)
        try:
            with key_path.open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            return self._proposal_cursor_key(index, create=False)
        return value

    @staticmethod
    def _complete_proposal_counts(
        proposal: CompleteProposal,
    ) -> dict[str, int]:
        counts = {
            "production_batches": (
                len(proposal.game_design.get("production_outline", ()))
                if isinstance(
                    proposal.game_design.get("production_outline"),
                    list,
                )
                else 0
            ),
            "modules": len(proposal.modules),
            "assets": len(proposal.assets),
            "audio": len(proposal.audio),
            "acceptance_tests": len(proposal.acceptance_tests),
        }
        contract = proposal.game_design.get("_production_contract")
        if isinstance(contract, dict) and isinstance(
            contract.get("catalog_stats"), dict
        ):
            stats = contract["catalog_stats"]
            counts["requirements"] = int(stats.get("requirements", 0))
            counts["quality_dimensions"] = int(
                stats.get("quality_dimensions", 0)
            )
        return counts

    def _new_child(self, relative: str) -> Path:
        target = self._resolve_child(relative)
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")
        target.mkdir(parents=True, exist_ok=False)
        return target

    def _new_file(self, relative: str) -> Path:
        target = self._resolve_child(relative)
        if target.exists():
            raise FileExistsError(f"Output already exists: {target}")
        return target

    def _existing_file(self, value: str) -> Path:
        path = self._resolve_child(value)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"File not found inside workspace: {path}"
            )
        return path

    def _existing_dir(self, value: str) -> Path:
        path = self._resolve_child(value)
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(
                f"Directory not found inside workspace: {path}"
            )
        return path

    def _scoped_media_paths(
        self,
        values: Sequence[str],
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise SpecValidationError(
                "media_paths must be a list of workspace files."
            )
        return tuple(
            str(self._existing_file(str(value)))
            for value in values
        )

    def _resolve_child(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise SpecValidationError(
                "Tool path escaped the configured workspace."
            ) from exc
        if target == self.workspace_root:
            raise SpecValidationError(
                "Tools may not target the workspace root itself."
            )
        return target


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecValidationError(
        "Model output did not contain a JSON object."
    )


def _minecraft_texture(source: Path, target: Path) -> None:
    from PIL import Image

    with Image.open(source) as image:
        image = image.convert("RGBA")
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        image = image.resize((16, 16), Image.Resampling.LANCZOS)
        image.save(target, format="PNG", optimize=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"
