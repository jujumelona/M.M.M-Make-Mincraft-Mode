from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Callable, Sequence

from .broker import LocalPolicyBroker, ToolAction, approved_request
from .complete_orchestrator import CompleteExecutionOptions, CompleteProductionOrchestrator
from .complete_planner import CompleteGameDesignPlanner
from .complete_spec import CompleteProposal
from .game_design import GameDesignPlanner
from .importer import inspect_existing_project_archive
from .knowledge import AuthoritativeEvidenceRetriever
from .model_router import ModelRouter
from .pipeline import MinecraftModPipeline
from .repair_engine import RepairEngine
from .runner import GradleRunner
from .source_patch import TransactionalSourcePatcher
from .spec import Proposal, ProposalStatus, SpecValidationError, canonical_json
from .validator import ProjectValidator, validate_jar


_SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class MMMToolService:
    """Concrete tool service used by the stdio MCP server and compatibility gateway."""

    def __init__(
        self,
        *,
        workspace_root: str | Path = "mmm-output",
        profile: str = "t4_local",
        router_factory: Callable[[], ModelRouter] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        self.router_factory = router_factory or (lambda: ModelRouter(profile=profile))
        self.broker = LocalPolicyBroker()

    def plan_game(
        self,
        prompt: str,
        media_paths: Sequence[str] = (),
    ) -> dict[str, Any]:
        planner = GameDesignPlanner(self.router_factory())
        design, proposal = planner.plan(prompt, media_paths=media_paths)
        return {
            "schema_version": "mmm/plan-result-v1",
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
        """Plan every requested gameplay, asset, world and runtime module."""
        proposal = CompleteGameDesignPlanner(self.router_factory()).plan(
            prompt,
            media_paths=media_paths,
            existing_input_sha256=existing_input_sha256,
        )
        return {
            "schema_version": "mmm/complete-plan-result-v1",
            "profile": self.profile,
            "game_design": proposal.game_design,
            "complete_proposal": proposal.to_dict(),
            "approval_hash": proposal.calculate_hash(),
        }

    def approve_complete_plan(
        self,
        complete_proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        parsed = CompleteProposal.from_dict(complete_proposal)
        approved = parsed.approve(approval_hash)
        return {
            "status": approved.status.value,
            "complete_proposal": approved.to_dict(),
            "approval_hash": approved.calculate_hash(),
        }

    def execute_complete_project(
        self,
        complete_proposal: dict[str, Any],
        approval_hash: str,
        run_name: str,
        options: dict[str, Any] | None = None,
        existing_input: str | None = None,
    ) -> dict[str, Any]:
        parsed_options = CompleteExecutionOptions(**(options or {}))
        return CompleteProductionOrchestrator(
            workspace_root=self.workspace_root,
            profile=self.profile,
            router_factory=self.router_factory,
        ).execute(
            complete_proposal,
            approval_hash=approval_hash,
            run_name=run_name,
            options=parsed_options,
            existing_input=existing_input,
        ).to_dict()

    def apply_source_patch(
        self,
        project_root: str,
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply one transactional, hash-guarded source patch."""
        root = self._existing_dir(project_root)
        return TransactionalSourcePatcher(root).apply(operations)

    def repair_project(
        self,
        project_root: str,
        run_gametest: bool = True,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        root = self._existing_dir(project_root)
        return RepairEngine(
            router=self.router_factory(),
            gradle_cache=self.workspace_root / ".cache" / "gradle",
        ).repair(root, run_gametest=run_gametest, max_attempts=max_attempts)

    def revise_plan(
        self,
        original_prompt: str,
        revision: str,
        media_paths: Sequence[str] = (),
    ) -> dict[str, Any]:
        if not revision.strip():
            raise SpecValidationError("revision must not be empty.")
        merged = f"{original_prompt.strip()}\n\nUser revision:\n{revision.strip()}"
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
        result = MinecraftModPipeline().execute(
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
        if not 1 <= len(assets) <= 16:
            raise SpecValidationError("assets must contain 1-16 id-to-prompt entries.")
        target = self._new_child(output_dir)
        concepts = target / "concepts"
        textures = target / "textures"
        concepts.mkdir(parents=True, exist_ok=True)
        textures.mkdir(parents=True, exist_ok=True)
        router = self.router_factory()
        generated: list[dict[str, str]] = []
        for index, (asset_id, prompt) in enumerate(sorted(assets.items())):
            if not _SAFE_ID.fullmatch(asset_id):
                raise SpecValidationError(f"Invalid asset id: {asset_id!r}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise SpecValidationError(f"Asset prompt is empty: {asset_id}")
            concept = router.generate_image(
                "image_generator",
                prompt=(
                    "Minecraft Java resource-pack source art, centered object, clean silhouette, "
                    "no text, no watermark, square composition. " + prompt.strip()
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
            "schema_version": "mmm/asset-result-v1",
            "output_dir": str(target),
            "generated": generated,
            "warning": "Generated textures require VisualCritic review before release.",
        }

    def generate_world_ir(
        self,
        brief: str,
        output_path: str = "world/world-ir.json",
        media_paths: Sequence[str] = (),
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return exactly one JSON object for a Minecraft 1.20.1 world design IR. "
                    "Keys: schema_version, regions, routes, structures, quests, constraints. "
                    "regions is a list of {id,purpose,biome_hint,entry_level}; routes is a list "
                    "of {from,to,travel_mode}; structures is a list of {id,region_id,kind,brief}; "
                    "quests is a list of {id,start_region,end_region,objective}. Use snake_case IDs. "
                    "This is planning IR only; do not claim NBT, Jigsaw or world files were generated."
                ),
            },
            {"role": "user", "content": brief},
        ]
        text = self.router_factory().generate_text(
            "world_planner", messages, media_paths=media_paths, response_format="json"
        )
        ir = _extract_json(text)
        _validate_world_ir(ir)
        target = self._new_file(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(canonical_json(ir) + "\n", encoding="utf-8")
        return {
            "schema_version": "mmm/world-ir-result-v1",
            "world_ir": ir,
            "path": str(target),
            "sha256": _sha256(target),
            "compiler_status": "available",
            "compiler_tool": "compile_world_ir",
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
        return ProjectValidator().validate(root, approved.spec).to_dict()

    def run_gradle_build(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        return self._run_gradle(
            project_root, proposal, approval_hash, run_gametest=False
        )

    def run_gametest(
        self,
        project_root: str,
        proposal: dict[str, Any],
        approval_hash: str,
    ) -> dict[str, Any]:
        return self._run_gradle(
            project_root, proposal, approval_hash, run_gametest=True
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
        source_report = ProjectValidator().validate(root, approved.spec)
        if not source_report.passed:
            raise RuntimeError("Source validation failed; release package was not created.")
        jar: Path | None = None
        jar_report: dict[str, Any] | None = None
        if jar_path is not None:
            jar = self._existing_file(jar_path)
            validated = validate_jar(jar, approved.spec)
            if not validated.passed:
                raise RuntimeError("JAR validation failed; release package was not created.")
            jar_report = validated.to_dict()
        target = self._new_file(output_zip)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Release already exists: {target}")
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(root)
                if any(part in {".gradle", ".cache", "gradle-user-home"} for part in relative.parts):
                    continue
                zipped.write(path, Path("source") / relative)
            if jar is not None:
                zipped.write(jar, Path("binary") / jar.name)
            zipped.writestr(
                "release-manifest.json",
                canonical_json(
                    {
                        "schema_version": "mmm/release-manifest-v1",
                        "proposal_hash": approved.calculate_hash(),
                        "source_validation": source_report.to_dict(),
                        "jar_validation": jar_report,
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
        action = ToolAction.GAME_TEST if run_gametest else ToolAction.GRADLE_BUILD
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
        return GradleRunner(cache).build(root, run_gametest=run_gametest).to_dict()

    @staticmethod
    def _approved(proposal: dict[str, Any], approval_hash: str) -> Proposal:
        parsed = Proposal.from_dict(proposal)
        approved = parsed.approve(approval_hash)
        if approved.status is not ProposalStatus.APPROVED:
            raise SpecValidationError("Proposal approval failed.")
        return approved

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
            raise FileNotFoundError(f"File not found inside workspace: {path}")
        return path

    def _existing_dir(self, value: str) -> Path:
        path = self._resolve_child(value)
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(f"Directory not found inside workspace: {path}")
        return path

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
            raise SpecValidationError("Tool path escaped the configured workspace.") from exc
        if target == self.workspace_root:
            raise SpecValidationError("Tools may not target the workspace root itself.")
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
    raise SpecValidationError("Model output did not contain a JSON object.")


def _validate_world_ir(ir: dict[str, Any]) -> None:
    required = {"schema_version", "regions", "routes", "structures", "quests", "constraints"}
    if set(ir) != required or ir.get("schema_version") != "mmm/world-ir-v1":
        raise SpecValidationError("World IR schema is invalid.")
    for key in ("regions", "routes", "structures", "quests", "constraints"):
        if not isinstance(ir[key], list):
            raise SpecValidationError(f"World IR field {key!r} must be a list.")
    region_ids: set[str] = set()
    for region in ir["regions"]:
        if not isinstance(region, dict) or not _SAFE_ID.fullmatch(str(region.get("id", ""))):
            raise SpecValidationError("World IR contains an invalid region id.")
        if region["id"] in region_ids:
            raise SpecValidationError(f"Duplicate region id: {region['id']}")
        region_ids.add(region["id"])
    for route in ir["routes"]:
        if not isinstance(route, dict) or route.get("from") not in region_ids or route.get("to") not in region_ids:
            raise SpecValidationError("World IR route references an unknown region.")


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
