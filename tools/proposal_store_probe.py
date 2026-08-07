from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from minecraft_mod_ai.complete_spec import (
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.proposal_store import (
    load_sharded_complete_proposal,
    load_sharded_complete_proposal_from_zip,
    write_sharded_complete_proposal,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "PROPOSAL_STORE_PROBE.json"


def snapshot(label: str, proposal) -> dict[str, object]:
    return {
        "label": label,
        "keys": sorted(proposal.game_design),
        "title": proposal.game_design.get("title"),
        "lore_chars": len(str(proposal.game_design.get("lore", ""))),
        "calculate_hash": proposal.calculate_hash(),
        "approval_hash": proposal.approval_hash,
    }


def main() -> None:
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan(
        "Create exactly one frost item."
    )
    source_design = {"title": "ZIP", "lore": "z" * 70_000}
    proposal = complete_proposal_from_parts(
        requested_prompt="Create a large deterministic content pack.",
        base_proposal=base,
        game_design=source_design,
        modules=tuple(
            ProductionModule(f"module_{index:05d}", "item")
            for index in range(19)
        ),
        acceptance_tests=("Every requested module is registered.",),
    )
    proposal = proposal.approve(proposal.calculate_hash())
    observations = [
        {
            "label": "source_design_after_proposal",
            "keys": sorted(source_design),
            "title": source_design.get("title"),
        },
        snapshot("proposal_before_write", proposal),
    ]

    with tempfile.TemporaryDirectory(prefix="mmm-proposal-probe-") as temp:
        tree = Path(temp) / "tree"
        index = write_sharded_complete_proposal(
            proposal,
            tree / "META-INF/mmm-complete-proposal.json",
            shard_size=3,
            part_size_bytes=16 * 1024,
        )
        observations.append(snapshot("proposal_after_write", proposal))
        loaded_file = load_sharded_complete_proposal(index)
        observations.append(snapshot("loaded_from_files", loaded_file))
        observations.append(snapshot("proposal_after_file_load", proposal))

        archive_path = Path(temp) / "proposal.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in sorted(tree.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(tree).as_posix())
        with zipfile.ZipFile(archive_path) as archive:
            loaded_zip = load_sharded_complete_proposal_from_zip(
                archive,
                index.relative_to(tree).as_posix(),
            )
        observations.append(snapshot("loaded_from_zip", loaded_zip))
        observations.append(snapshot("proposal_after_zip_load", proposal))
        observations.append(
            {
                "label": "equality",
                "file_equals_proposal": (
                    loaded_file.game_design == proposal.game_design
                ),
                "zip_equals_proposal": (
                    loaded_zip.game_design == proposal.game_design
                ),
                "file_equals_zip": (
                    loaded_file.game_design == loaded_zip.game_design
                ),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(observations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
