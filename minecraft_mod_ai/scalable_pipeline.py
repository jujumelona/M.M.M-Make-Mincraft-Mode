from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .importer import inspect_existing_project_archive
from .pipeline import MinecraftModPipeline
from .platform_resolver import retarget_proposal
from .platform_selection_pipeline import resolve_platform_fail_closed
from .scalable_generator import ScalableFabricProjectGenerator
from .scalable_validator import ScalableProjectValidator
from .scale_policy import ScalePolicy


class ScalableMinecraftModPipeline(MinecraftModPipeline):
    """Scalable pipeline with explicit fail-closed planning and deterministic execution."""

    def __init__(self, *, planner=None, broker=None, policy: ScalePolicy | None = None) -> None:
        super().__init__(planner=planner, broker=broker)
        self.policy = policy or ScalePolicy.from_environment()
        self.policy.validate()
        self.generator = ScalableFabricProjectGenerator(policy=self.policy)
        self.validator = ScalableProjectValidator(policy=self.policy)

    def plan(
        self,
        prompt: str,
        *,
        existing_input: str | Path | None = None,
    ):
        """Create one target-bound proposal without serializing independent input work."""

        report = None
        if existing_input is None:
            proposal = self.planner.plan(prompt)
        else:
            # Planner inference and archive inspection consume independent inputs.
            # Run them concurrently, then join only at platform resolution where
            # both results are actually required.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="mmm-plan") as pool:
                proposal_future = pool.submit(self.planner.plan, prompt)
                report_future = pool.submit(inspect_existing_project_archive, existing_input)
                proposal = proposal_future.result()
                report = report_future.result()

        requested_version = getattr(self.planner, "_mmm_requested_minecraft_version", None)
        requested_loader = getattr(self.planner, "_mmm_requested_loader", None)
        effective_prompt = str(prompt)
        if requested_version and str(requested_version) not in effective_prompt:
            effective_prompt += f"\n[HOST_TARGET_CONSTRAINT Minecraft {requested_version}]"
        if requested_loader and str(requested_loader).casefold() not in effective_prompt.casefold():
            effective_prompt += f"\n[HOST_LOADER_CONSTRAINT {requested_loader}]"

        selection = resolve_platform_fail_closed(
            effective_prompt,
            existing_version=(report.minecraft_version if report is not None else None),
            existing_loader=(report.loader if report is not None else None),
        )
        proposal = retarget_proposal(proposal, selection)
        if report is not None:
            proposal = self._bind_existing_input(proposal, report)
        proposal.validate()
        return proposal
