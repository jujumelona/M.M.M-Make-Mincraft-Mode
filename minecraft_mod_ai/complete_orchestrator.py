from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .audio_generator import generate_audio_assets
from .blockbench_client import BlockbenchMCPClient
from .complete_spec import CompleteProposal, CompleteProposalStatus, ProductionModule
from .custom_module_generator import CustomModuleGenerator
from .extended_content_generator import generate_extended_content
from .generator import FabricProjectGenerator
from .geckolib_generator import generate_geckolib_entity_assets
from .java_lsp import JavaLanguageService
from .importer import inspect_existing_project_archive
from .mineflayer_bridge import MineflayerBridge
from .model_router import ModelRouter
from .pipeline import MinecraftModPipeline
from .project_edit import inspect_fabric_project
from .publisher import (
    build_distribution_metadata,
    package_distribution_bundle,
    publish_curseforge,
    publish_modrinth,
)
from .repair_engine import RepairEngine
from .runner import GradleRunner
from .runtime_manager import MinecraftRuntimeManager
from .source_patch import TransactionalSourcePatcher, sha256_file
from .spec import SpecValidationError
from .system_pack_generator import generate_system_pack
from .validator import ProjectValidator, validate_jar
from .world_compiler import compile_world_ir


class CompleteProductionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompleteExecutionOptions:
    source_only: bool = False
    run_jdt: bool = True
    run_gametest: bool = True
    auto_repair: bool = True
    max_repair_attempts: int = 3
    run_blockbench: bool = True
    run_runtime: bool = True
    run_client: bool = True
    run_mineflayer: bool = True
    run_visual_review: bool = True
    cleanup_runtime: bool = True
    eula_accepted: bool = False
    server_launcher: str | None = None
    screenshot_paths: tuple[str, ...] = ()
    playtest_actions: tuple[dict[str, Any], ...] = ()
    publish_provider: str | None = None
    publish_project_id: str | None = None
    changelog: str = "Generated and verified by M.M.M"

    def validate(self) -> None:
        if not 0 <= self.max_repair_attempts <= 5:
            raise CompleteProductionError("max_repair_attempts must be 0-5.")
        if self.publish_provider not in {None, "modrinth", "curseforge"}:
            raise CompleteProductionError("publish_provider must be modrinth or curseforge.")
        if self.publish_provider and not self.publish_project_id:
            raise CompleteProductionError("publish_project_id is required when publishing.")
        if self.source_only and self.publish_provider:
            raise CompleteProductionError("Source-only execution cannot publish a binary release.")


@dataclass(frozen=True)
class CompletePipelineResult:
    schema_version: str
    status: str
    project_root: str
    release_zip: str | None
    jar_path: str | None
    complete_proposal_hash: str
    source_validation: dict[str, Any]
    build_report: dict[str, Any] | None
    jar_validation: dict[str, Any] | None
    module_receipts: tuple[dict[str, Any], ...]
    world_receipt: dict[str, Any] | None
    asset_receipt: dict[str, Any] | None
    audio_receipt: dict[str, Any] | None
    blockbench_receipts: tuple[dict[str, Any], ...]
    runtime_receipt: dict[str, Any] | None
    playtest_receipt: dict[str, Any] | None
    visual_receipt: dict[str, Any] | None
    distribution_receipt: dict[str, Any] | None
    unresolved_gates: tuple[str, ...]
    release_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompleteProductionOrchestrator:
    """One approved request -> source -> repair -> runtime -> release orchestration."""

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

    def execute(
        self,
        proposal: CompleteProposal | dict[str, Any],
        *,
        approval_hash: str,
        run_name: str,
        options: CompleteExecutionOptions | None = None,
        existing_input: str | Path | None = None,
    ) -> CompletePipelineResult:
        options = options or CompleteExecutionOptions()
        options.validate()
        parsed = proposal if isinstance(proposal, CompleteProposal) else CompleteProposal.from_dict(proposal)
        approved = parsed.approve(approval_hash)
        if approved.status is not CompleteProposalStatus.APPROVED:
            raise SpecValidationError("Complete proposal approval did not complete.")
        run_root = self._new_run_root(run_name)
        router: ModelRouter | None = None
        module_receipts: list[dict[str, Any]] = []
        blockbench_receipts: list[dict[str, Any]] = []
        world_receipt: dict[str, Any] | None = None
        asset_receipt: dict[str, Any] | None = None
        audio_receipt: dict[str, Any] | None = None
        runtime_receipt: dict[str, Any] | None = None
        playtest_receipt: dict[str, Any] | None = None
        visual_receipt: dict[str, Any] | None = None
        distribution_receipt: dict[str, Any] | None = None
        unresolved: list[str] = []

        project_root = self._prepare_project(
            approved,
            run_root=run_root,
            existing_input=existing_input,
        )
        self._write_complete_approval(project_root, approved)
        base = approved.base_proposal
        spec = base.spec
        ordered = _topological_modules(approved.modules)

        # Deterministic built-in content and actual Fabric system bindings.
        extended = generate_extended_content(
            project_root=project_root,
            mod_id=spec.mod_id,
            package_name=spec.package_name,
            modules=ordered,
        )
        if extended.get("status") != "SKIPPED":
            module_receipts.append(extended)

        for pack_id, pack_modules in _system_groups(ordered).items():
            module_receipts.append(
                generate_system_pack(
                    project_root=project_root,
                    pack_id=pack_id,
                    mod_id=spec.mod_id,
                    package_name=spec.package_name,
                    config={"modules": [_module_dict(item) for item in pack_modules]},
                )
            )

        entity_modules = [item for item in ordered if item.kind in {"entity", "boss", "npc"}]
        for module in entity_modules:
            config = module.config
            receipt = generate_geckolib_entity_assets(
                project_root=project_root,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                entity_id=module.module_id,
                max_health=float(config.get("max_health", 80.0)),
                attack_damage=float(config.get("attack_damage", 8.0)),
                movement_speed=float(config.get("movement_speed", 0.27)),
            )
            module_receipts.append(receipt)
            if options.run_blockbench and not options.source_only:
                blockbench_receipts.append(self._blockbench_review(receipt, run_root))
            elif options.run_blockbench:
                unresolved.append(f"blockbench:{module.module_id}:not-run-in-source-only-mode")

        if approved.world_ir is not None:
            world_output = run_root / "world-compiled"
            world_receipt = compile_world_ir(
                approved.world_ir,
                mod_id=spec.mod_id,
                output_root=world_output,
                package_world_zip=True,
            )
            self._merge_world_resources(world_output, project_root, spec.mod_id)

        if approved.assets:
            router = router or self.router_factory()
            asset_receipt = self._generate_assets(router, approved, project_root, run_root)

        if approved.audio:
            audio_receipt = generate_audio_assets(
                project_root=project_root,
                mod_id=spec.mod_id,
                package_name=spec.package_name,
                requests=approved.audio,
            )

        handled = _handled_module_ids(ordered)
        if approved.world_ir is None:
            handled -= {module.module_id for module in ordered if module.kind == "structure"}
        if not approved.audio:
            handled -= {module.module_id for module in ordered if module.kind == "audio"}
        custom_modules = [module for module in ordered if module.module_id not in handled]
        for module in custom_modules:
            router = router or self.router_factory()
            module_receipts.append(
                CustomModuleGenerator(router).generate(project_root, module=module)
            )

        source_report = ProjectValidator().validate(project_root, spec).to_dict()
        if source_report.get("status") != "PASS":
            raise CompleteProductionError("Generated complete project failed deterministic validation.")
        if options.run_jdt and not options.source_only:
            try:
                jdt_receipt = JavaLanguageService().diagnostics(project_root, timeout_seconds=90)
            except Exception as exc:
                jdt_receipt = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
            module_receipts.append({"schema_version": "mmm/jdt-gate-v1", **jdt_receipt})
            if jdt_receipt.get("status") == "UNAVAILABLE":
                raise CompleteProductionError(
                    "JDT Language Server is required for a fully verified build: "
                    + str(jdt_receipt.get("error", "unavailable"))
                )
            errors = [
                item for item in jdt_receipt.get("diagnostics", [])
                if isinstance(item, dict) and int(item.get("severity", 1)) <= 2
            ]
            if errors and not options.auto_repair:
                raise CompleteProductionError("JDT reported errors and automatic repair is disabled.")

        if options.source_only:
            release = self._package_source_only(
                run_root,
                project_root,
                approved,
            )
            unresolved.extend(_external_gates(approved, options))
            return CompletePipelineResult(
                schema_version="mmm/complete-pipeline-result-v1",
                status="SOURCE_READY",
                project_root=str(project_root),
                release_zip=release,
                jar_path=None,
                complete_proposal_hash=approved.calculate_hash(),
                source_validation=source_report,
                build_report=None,
                jar_validation=None,
                module_receipts=tuple(module_receipts),
                world_receipt=world_receipt,
                asset_receipt=asset_receipt,
                audio_receipt=audio_receipt,
                blockbench_receipts=tuple(blockbench_receipts),
                runtime_receipt=None,
                playtest_receipt=None,
                visual_receipt=None,
                distribution_receipt=None,
                unresolved_gates=tuple(sorted(set(unresolved))),
                release_ready=False,
            )

        cache = run_root / ".cache/gradle"
        build = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
        if build.get("status") != "PASS" and options.auto_repair and options.max_repair_attempts:
            router = router or self.router_factory()
            repair = RepairEngine(router=router, gradle_cache=cache).repair(
                project_root,
                run_gametest=options.run_gametest,
                max_attempts=options.max_repair_attempts,
            )
            module_receipts.append({"schema_version": "mmm/repair-receipt-v1", **repair})
            build = GradleRunner(cache).build(project_root, run_gametest=options.run_gametest).to_dict()
        if build.get("status") != "PASS":
            raise CompleteProductionError("Gradle/GameTest failed after the bounded repair loop.")
        jar_path = _jar_path(build)
        jar_validation = validate_jar(jar_path, spec).to_dict()
        if jar_validation.get("status") != "PASS":
            raise CompleteProductionError("Built JAR failed independent validation.")

        runtime_manager: MinecraftRuntimeManager | None = None
        try:
            if options.run_runtime:
                if not options.server_launcher:
                    raise CompleteProductionError("server_launcher is required for complete runtime verification.")
                if not options.eula_accepted:
                    raise CompleteProductionError("Explicit Minecraft EULA acceptance is required.")
                runtime_manager = MinecraftRuntimeManager(run_root)
                launcher_source = Path(options.server_launcher).expanduser().resolve()
                if not launcher_source.is_file() or launcher_source.is_symlink():
                    raise CompleteProductionError("server_launcher must be a regular file.")
                launcher_copy = run_root / "integration-inputs/fabric-server-launch.jar"
                launcher_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(launcher_source, launcher_copy)
                prepared = runtime_manager.prepare_instance(
                    "complete-integration",
                    mod_jar=jar_path,
                    server_launcher=launcher_copy,
                    eula_accepted=True,
                )
                server = runtime_manager.start_server(timeout_seconds=180)
                client = runtime_manager.start_client() if options.run_client else None
                runtime_receipt = {"prepared": prepared, "server": server, "client": client}
            else:
                unresolved.append("runtime:not-requested")

            if options.run_mineflayer:
                if not options.run_runtime:
                    raise CompleteProductionError("Mineflayer requires the disposable runtime.")
                playtest_receipt = self._run_playtest(options.playtest_actions)
            else:
                unresolved.append("mineflayer:not-requested")

            if options.run_visual_review:
                if not options.screenshot_paths:
                    raise CompleteProductionError(
                        "Visual review requires explicit runtime screenshot paths."
                    )
                router = router or self.router_factory()
                visual_receipt = self._visual_review(router, approved, options.screenshot_paths)
                if visual_receipt.get("status") != "PASS":
                    raise CompleteProductionError("VisualCritic rejected the runtime screenshots.")
            else:
                unresolved.append("visual-review:not-requested")
        finally:
            if runtime_manager is not None and options.cleanup_runtime:
                cleanup = runtime_manager.cleanup()
                runtime_receipt = {**(runtime_receipt or {}), "cleanup": cleanup}

        from .mcp_tools import MMMToolService

        tool_service = MMMToolService(workspace_root=run_root, profile=self.profile)
        release_result = tool_service.package_release(
            str(project_root),
            base.to_dict(),
            base.calculate_hash(),
            output_zip="releases/complete-release.zip",
            jar_path=str(jar_path),
        )
        release_zip = str(release_result["release_zip"])

        metadata = build_distribution_metadata(
            jar_path=jar_path,
            mod_id=spec.mod_id,
            version=spec.version,
            name=spec.mod_name,
            changelog=options.changelog,
        )
        bundle = package_distribution_bundle(
            metadata,
            output_zip=run_root / "releases/distribution-bundle.zip",
            source_zip=release_zip,
        )
        distribution_receipt = {"metadata": metadata, "bundle": bundle}
        if options.publish_provider == "modrinth":
            distribution_receipt["publish"] = publish_modrinth(
                metadata,
                project_id=str(options.publish_project_id),
            )
        elif options.publish_provider == "curseforge":
            distribution_receipt["publish"] = publish_curseforge(
                metadata,
                project_id=str(options.publish_project_id),
            )

        release_ready = not unresolved
        return CompletePipelineResult(
            schema_version="mmm/complete-pipeline-result-v1",
            status="VERIFIED" if release_ready else "BUILT_WITH_UNRESOLVED_GATES",
            project_root=str(project_root),
            release_zip=release_zip,
            jar_path=str(jar_path),
            complete_proposal_hash=approved.calculate_hash(),
            source_validation=source_report,
            build_report=build,
            jar_validation=jar_validation,
            module_receipts=tuple(module_receipts),
            world_receipt=world_receipt,
            asset_receipt=asset_receipt,
            audio_receipt=audio_receipt,
            blockbench_receipts=tuple(blockbench_receipts),
            runtime_receipt=runtime_receipt,
            playtest_receipt=playtest_receipt,
            visual_receipt=visual_receipt,
            distribution_receipt=distribution_receipt,
            unresolved_gates=tuple(sorted(set(unresolved))),
            release_ready=release_ready,
        )

    def _prepare_project(
        self,
        approved: CompleteProposal,
        *,
        run_root: Path,
        existing_input: str | Path | None,
    ) -> Path:
        if existing_input is not None:
            report = inspect_existing_project_archive(
                existing_input,
                extract_root=run_root / "existing-source",
            )
            if approved.existing_input_sha256 and report.archive_sha256 != approved.existing_input_sha256:
                raise CompleteProductionError("Existing input bytes changed after complete-plan approval.")
            if not report.has_sources or not report.has_gradle_project or not report.extracted_to:
                raise CompleteProductionError("Existing mod modification requires a source Gradle ZIP.")
            project_root = _locate_existing_fabric_root(Path(report.extracted_to).resolve())
            info = inspect_fabric_project(project_root)
            expected = approved.base_proposal.spec
            if info.mod_id != expected.mod_id or info.package_name != expected.package_name:
                raise CompleteProductionError(
                    "Approved proposal does not match the existing Fabric project: "
                    f"expected {expected.mod_id}/{expected.package_name}, "
                    f"found {info.mod_id}/{info.package_name}. Re-plan against the inspected source."
                )
            return project_root
        base = approved.base_proposal
        base.approve(base.calculate_hash())
        project_root = run_root / "base/workspaces" / base.spec.mod_id
        FabricProjectGenerator().generate(base.spec, project_root)
        metadata = project_root / ".minecraft_ai"
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "base-proposal.json").write_text(
            json.dumps(base.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (metadata / "complete-proposal.json").write_text(
            json.dumps(approved.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return project_root.resolve()

    @staticmethod
    def _write_complete_approval(project_root: Path, proposal: CompleteProposal) -> None:
        rendered = json.dumps(
            proposal.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        audit = project_root / ".minecraft_ai/complete-proposal.json"
        embedded = project_root / "src/main/resources/META-INF/mmm-complete-proposal.json"
        for path in (audit, embedded):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")

    def _new_run_root(self, run_name: str) -> Path:
        if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in run_name):
            raise CompleteProductionError("run_name must use lowercase letters, numbers, underscore or hyphen.")
        root = (self.workspace_root / run_name).resolve()
        try:
            root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise CompleteProductionError("Run path escaped the workspace.") from exc
        if root.exists():
            raise FileExistsError(root)
        root.mkdir(parents=True)
        return root

    def _merge_world_resources(self, world_output: Path, project_root: Path, mod_id: str) -> None:
        source = world_output / "data" / mod_id
        target = project_root / "src/main/resources/data" / mod_id
        if not source.is_dir():
            raise CompleteProductionError("World compiler did not produce the expected namespace.")
        shutil.copytree(source, target, dirs_exist_ok=True)
        manifest = world_output / "mmm-world-manifest.json"
        shutil.copy2(manifest, project_root / ".minecraft_ai/mmm-world-manifest.json")

    def _generate_assets(
        self,
        router: ModelRouter,
        proposal: CompleteProposal,
        project_root: Path,
        run_root: Path,
    ) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise CompleteProductionError("Pillow is required for texture post-processing.") from exc
        generated: list[dict[str, Any]] = []
        concept_dir = run_root / "asset-concepts"
        concept_dir.mkdir(parents=True, exist_ok=True)
        for index, request in enumerate(proposal.assets):
            concept = router.generate_image(
                "image_generator",
                prompt=(
                    "Minecraft Java texture source, centered, clean silhouette, no text, no watermark. "
                    + request.prompt
                ),
                output_path=concept_dir / f"{request.asset_id}.png",
                width=512,
                height=512,
                seed=index,
            )
            target = (project_root / request.target_path).resolve()
            try:
                target.relative_to(project_root)
            except ValueError as exc:
                raise CompleteProductionError("Asset target escaped the project root.") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(concept) as image:
                image.convert("RGBA").resize((request.width, request.height), Image.Resampling.NEAREST).save(target)
            generated.append(
                {
                    "asset_id": request.asset_id,
                    "concept": str(concept),
                    "target": str(target),
                    "sha256": sha256_file(target),
                }
            )
        return {"schema_version": "mmm/complete-assets-v1", "status": "GENERATED", "assets": generated}

    def _blockbench_review(self, gecko_receipt: dict[str, Any], run_root: Path) -> dict[str, Any]:
        geo = next((path for path in gecko_receipt.get("files", []) if str(path).endswith(".geo.json")), None)
        if not geo:
            raise CompleteProductionError("GeckoLib receipt did not contain geometry.")
        preview = run_root / "blockbench-previews" / (Path(geo).stem + ".png")
        preview.parent.mkdir(parents=True, exist_ok=True)
        client = BlockbenchMCPClient()
        try:
            client.call("open_project", {"path": geo})
            uv = client.call("validate_uv", {})
            render = client.call("render_preview", {"output_path": str(preview)})
            client.call("close_project", {})
        finally:
            client.close()
        return {"entity": gecko_receipt["entity_id"], "uv": uv, "render": render, "preview": str(preview)}

    def _run_playtest(self, actions: Iterable[dict[str, Any]]) -> dict[str, Any]:
        bridge = MineflayerBridge()
        results: list[dict[str, Any]] = []
        try:
            results.append(bridge.call("connect", host="127.0.0.1", port=25565, username="MMMTestBot"))
            for action in actions:
                if not isinstance(action, dict) or "action" not in action:
                    raise CompleteProductionError("Every playtest action must contain action.")
                name = str(action["action"])
                if name not in {"status", "walk_to", "interact_block", "use_item", "attack_entity", "inventory"}:
                    raise CompleteProductionError(f"Unsupported playtest action: {name}")
                params = dict(action.get("params", {}))
                results.append(bridge.call(name, **params))
            results.append(bridge.call("inventory"))
            return {"schema_version": "mmm/playtest-result-v1", "status": "PASS", "results": results}
        finally:
            try:
                bridge.call("disconnect")
            except Exception:
                pass

    def _visual_review(
        self,
        router: ModelRouter,
        proposal: CompleteProposal,
        screenshots: tuple[str, ...],
    ) -> dict[str, Any]:
        paths = [Path(value).expanduser().resolve() for value in screenshots]
        if any(not path.is_file() or path.is_symlink() for path in paths):
            raise CompleteProductionError("Every visual-review screenshot must be a regular file.")
        text = router.generate_text(
            "visual_critic",
            [
                {
                    "role": "system",
                    "content": (
                        "Return JSON {status: PASS|FAIL, findings: [...], acceptance_test_results: [...]} "
                        "for Minecraft runtime screenshots. Reject missing textures, broken models, unreadable GUI, "
                        "animation clipping and deviations from the approved design."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "game_design": proposal.game_design,
                            "acceptance_tests": list(proposal.acceptance_tests),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            media_paths=paths,
            response_format="json",
        )
        value = _extract_json(text)
        if value.get("status") not in {"PASS", "FAIL"} or not isinstance(value.get("findings"), list):
            raise CompleteProductionError("VisualCritic returned an invalid result contract.")
        return {"schema_version": "mmm/visual-review-v1", **value, "screenshots": [str(path) for path in paths]}

    def _package_source_only(
        self,
        run_root: Path,
        project_root: Path,
        proposal: CompleteProposal,
    ) -> str:
        target = run_root / "releases/complete-source.zip"
        target.parent.mkdir(parents=True, exist_ok=True)
        import zipfile

        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(project_root.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(project_root)
                if any(part in {".gradle", "build", "run", ".cache"} for part in relative.parts):
                    continue
                archive.write(path, Path("source") / relative)
            archive.writestr(
                "complete-proposal.json",
                json.dumps(proposal.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        return str(target)



def _locate_existing_fabric_root(extracted_root: Path) -> Path:
    direct = extracted_root / "src/main/resources/fabric.mod.json"
    if direct.is_file() and not direct.is_symlink():
        return extracted_root
    candidates = sorted(
        path.parent.parent.parent.parent
        for path in extracted_root.rglob("fabric.mod.json")
        if path.as_posix().endswith("src/main/resources/fabric.mod.json")
        and path.is_file()
        and not path.is_symlink()
    )
    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(extracted_root)
        except ValueError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if len(unique) != 1:
        raise CompleteProductionError(
            "Existing source ZIP must contain exactly one Fabric project root; "
            f"found {len(unique)}."
        )
    return unique[0]

def _topological_modules(modules: tuple[ProductionModule, ...]) -> list[ProductionModule]:
    lookup = {module.module_id: module for module in modules}
    pending = set(lookup)
    ordered: list[ProductionModule] = []
    emitted: set[str] = set()
    while pending:
        ready = sorted(module_id for module_id in pending if set(lookup[module_id].depends_on) <= emitted)
        if not ready:
            raise CompleteProductionError("Production module graph contains an unresolved cycle.")
        for module_id in ready:
            ordered.append(lookup[module_id])
            emitted.add(module_id)
            pending.remove(module_id)
    return ordered


def _system_groups(modules: list[ProductionModule]) -> dict[str, list[ProductionModule]]:
    mapping = {
        "quest": "quest-system",
        "class": "class-skill-system",
        "skill": "class-skill-system",
        "economy": "economy-shop",
        "shop": "economy-shop",
        "gui": "gui-networking",
        "networking": "gui-networking",
        "party": "party-guild",
        "guild": "party-guild",
    }
    result: dict[str, list[ProductionModule]] = {}
    for module in modules:
        pack = mapping.get(module.kind)
        if pack:
            result.setdefault(pack, []).append(module)
    return result


def _handled_module_ids(modules: list[ProductionModule]) -> set[str]:
    built_in = {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "machine",
        "effect",
        "enchantment",
        "command",
        "recipe",
        "advancement",
        "loot",
        "quest",
        "class",
        "skill",
        "economy",
        "shop",
        "gui",
        "networking",
        "party",
        "guild",
        "entity",
        "boss",
        "structure",
        "audio",
    }
    # NPC receives GeckoLib source and then a custom interaction patch.
    return {module.module_id for module in modules if module.kind in built_in}


def _module_dict(module: ProductionModule) -> dict[str, Any]:
    return {
        "module_id": module.module_id,
        "kind": module.kind,
        "config": module.config,
        "depends_on": list(module.depends_on),
        "required_gates": list(module.required_gates),
    }


def _jar_path(build: dict[str, Any]) -> Path:
    value = build.get("jar_path")
    if not isinstance(value, str):
        raise CompleteProductionError("Gradle report did not contain a JAR path.")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise CompleteProductionError("Gradle JAR path is missing or unsafe.")
    return path


def _external_gates(proposal: CompleteProposal, options: CompleteExecutionOptions) -> list[str]:
    gates = ["Gradle", "GameTest", "JAR validation"]
    if proposal.external_runtime_required:
        gates.extend(["Minecraft server/client runtime", "Mineflayer playtest", "visual review"])
    if any(module.kind in {"entity", "boss", "npc"} for module in proposal.modules):
        gates.append("Blockbench UV/render review")
    return gates


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
    raise CompleteProductionError("Model response did not contain a JSON object.")
