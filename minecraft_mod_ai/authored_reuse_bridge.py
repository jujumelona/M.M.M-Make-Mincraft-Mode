"""Host-owned source-reuse handoff for the default authored production lane.

Discovery candidates are never executable code. Only proven donor decisions
may be materialized into the final workspace. The small model gets a bounded
receipt, never authority to invent or select a donor.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def authored_capability_graph(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
) -> dict[str, Any]:
    from .minecraft_knowledge_nodes import detect_features

    # Known taxonomy only supplies search keywords, never the request's scope.
    # Include the actual authored request to preserve uncatalogued mechanics.
    features = [
        str(name).replace("_", " ")
        for name in detect_features(requested_prompt)
        if str(name) != "base_mod"
    ]
    full_request = " ".join(requested_prompt.split())

    # Do not search a Korean/free-form multi-feature request as one GitHub
    # keyword query. The approved structured plan already identifies concrete
    # systems and interactions in provider-searchable technical vocabulary.
    # These are *retrieval hints*, not newly invented gameplay requirements.
    import re

    hints: list[str] = []
    for section, collection, keys in (
        ("reuse_assessment", "adaptations", ("part",)),
        ("resources_and_ui", "assets", ("purpose", "production_owner")),
        ("integration", "entry_points", ("boundary",)),
        ("behavior_contract", "outputs", ("name",)),
    ):
        raw_section = structured_sections.get(section, {})
        spec = raw_section.get("specification", {}) if isinstance(raw_section, Mapping) else {}
        records = spec.get(collection, ()) if isinstance(spec, Mapping) else ()
        if not isinstance(records, (list, tuple)):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key in keys:
                phrase = re.sub(r"[_./-]+", " ", str(record.get(key) or "").casefold())
                phrase = " ".join(phrase.split())
                words = phrase.split()
                if (
                    1 <= len(words) <= 6
                    and len(phrase) <= 70
                    and phrase not in {"player", "host", "game system", "game"}
                    and phrase not in hints
                ):
                    hints.append(phrase)
    nodes = list(dict.fromkeys(
        ["custom gameplay " + full_request[:150]]
        + ["minecraft fabric " + name for name in features]
        + ["minecraft fabric " + hint for hint in hints]
    ))
    return {
        "nodes": nodes,
        "search_terms": [
            {
                "capability": node,
                "terms": (
                    [f"minecraft {node.removeprefix('minecraft fabric ')} mod", node]
                    if node.startswith("minecraft fabric ")
                    else [*("minecraft " + hint + " mod" for hint in hints), node]
                ),
            }
            for node in nodes
        ],
        "retrieval_policy_version": "structured-intent-v2",
        "coverage_policy": "discovery_hints_only; authored design remains authoritative",
    }


def resolve_authored_source_reuse(
    requested_prompt: str,
    structured_sections: Mapping[str, Any],
    *,
    minecraft_version: str,
    loader: str,
) -> dict[str, Any]:
    from .grounded_source_reuse import build_repository_reuse_plan

    graph = authored_capability_graph(requested_prompt, structured_sections)
    frozen = {
        "plan_sha256": "sha256:" + _sha(graph),
        "capability_graph": graph,
    }
    design = {
        "_pre_retrieval_plan": frozen,
        "_platform_selection": {
            "target": {
                "minecraft_version": minecraft_version,
                "loader": loader,
            },
        },
    }
    receipt = build_repository_reuse_plan(design)
    return {
        **receipt,
        "bound_target": {
            "minecraft_version": minecraft_version,
            "loader": loader,
        },
        "origin": "default_authored_production",
    }


def verified_reuse_context(plan: Mapping[str, Any]) -> str:
    """Compact, authority-preserving context for a low-memory local model."""
    decisions = plan.get("capabilities", ())
    summaries = []
    for row in decisions if isinstance(decisions, list) else ():
        if not isinstance(row, Mapping):
            continue
        source_id = str(row.get("source_id") or "")
        summaries.append({
            "capability": str(row.get("capability") or "")[:160],
            "source": source_id if row.get("mode") == "source_transplant" else "",
            "mode": row.get("mode", "fresh"),
        })
    return (
        "\n\nHOST SOURCE-REUSE RECEIPTS (read-only facts, not new requests):\n"
        + json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
        + "\nOnly source_transplant rows have passed code inspection and compile proof. "
        "Fresh rows need an independent implementation. Never treat a repository search "
        "result or a feature description as working code.\n"
    )


def _verified_donor_license_notice(
    donor: Mapping[str, Any],
    blobs: Mapping[str, bytes],
) -> tuple[str, bytes]:
    """Retain a pinned donor license notice; never invent an attribution."""
    license_text = next((
        content for path, content in blobs.items()
        if path.casefold().split("/")[-1] in {
            "license", "license.txt", "license.md",
            "copying", "copying.txt", "copying.md",
        }
    ), None)
    if license_text is None:
        import httpx

        repository = str(donor["repository"])
        commit = str(donor["commit_sha"])
        with httpx.Client(timeout=15, follow_redirects=True) as http:
            for filename in (
                "LICENSE", "LICENSE.md", "LICENSE.txt",
                "COPYING", "COPYING.md", "COPYING.txt",
            ):
                url = (
                    "https://raw.githubusercontent.com/"
                    + repository + "/" + commit + "/" + filename
                )
                try:
                    response = http.get(url)
                except httpx.HTTPError:
                    continue
                if response.status_code == 200 and 0 < len(response.content) < 256_000:
                    license_text = response.content
                    break
    if not license_text:
        raise RuntimeError(
            "SOURCE_REUSE_LICENSE_NOTICE_REQUIRED: " + str(donor["repository"])
        )
    filename = (
        str(donor["repository"]).replace("/", "__").replace(".", "_")
        + "-" + str(donor["commit_sha"])[:12] + ".LICENSE"
    )
    return "src/main/resources/META-INF/mmm-third-party/" + filename, license_text


def materialize_verified_authored_sources(
    project_root: str,
    reuse_plan: Mapping[str, Any],
    *,
    minecraft_version: str,
    loader: str,
    package_name: str = "",
    mod_id: str = "",
) -> dict[str, Any]:
    """Install only proof-bound donor source into the real production workspace.

    Reuse without source files in the build is not reuse. Donor code is
    hash-verified again when downloaded, deterministically adapted, and never
    permitted to overwrite an existing project's Java or resource files.
    """
    from pathlib import Path
    from .source_transplant import materialize_source_slices
    from .reuse_adapters import apply_deterministic_adapters

    bound = reuse_plan.get("bound_target")
    if bound != {"minecraft_version": minecraft_version, "loader": loader}:
        raise ValueError("SOURCE_REUSE_TARGET_MISMATCH: donor proof is for another target")

    chosen = [
        row for row in reuse_plan.get("capabilities", ())
        if isinstance(row, Mapping) and row.get("mode") == "source_transplant"
    ]
    if not chosen:
        return {"schema_version": "mmm/authored-source-install-v1", "donor_count": 0, "files": []}
    if not package_name.strip() or not mod_id.strip():
        raise ValueError(
            "SOURCE_REUSE_TARGET_IDENTITY_REQUIRED: verified transplant needs "
            "the actual mod package and mod ID for deterministic adaptation."
        )

    root = Path(project_root).resolve()
    downloaded = materialize_source_slices(root, reuse_plan)
    donors = downloaded.get("donors", ())
    if len(donors) != len(chosen):
        raise RuntimeError("SOURCE_REUSE_MATERIALIZATION_INCOMPLETE")

    # All files are inspected before writing any donor into the production tree.
    prepared: dict[str, tuple[bytes, str]] = {}
    supported = (
        "src/main/java/", "src/main/resources/",
        "src/client/java/", "src/client/resources/",
    )
    from .platform_catalog import adapter_for_target
    adapter = adapter_for_target(minecraft_version, loader)
    target_context = {
        "minecraft_version": adapter.minecraft_version,
        "loader": adapter.loader,
        "mappings": adapter.yarn_mappings,
        "mappings_kind": adapter.mappings_kind,
        "java_version": adapter.java_version,
        "fabric_loader": adapter.fabric_loader,
        "fabric_api": adapter.fabric_api,
        "fabric_loom": adapter.fabric_loom,
        "gradle": adapter.gradle,
        "target_package": package_name,
        "target_modid": mod_id,
    }
    attribution: list[dict[str, Any]] = []
    for chosen_row, donor in zip(chosen, donors):
        if donor.get("capability") != chosen_row.get("capability"):
            raise RuntimeError("SOURCE_REUSE_DONOR_RECEIPT_MISMATCH")
        dependencies = donor.get("required_dependencies") or ()
        if dependencies:
            # A compiled isolated donor does not automatically install its Maven
            # dependencies into the final project. Refuse a false success.
            raise RuntimeError(
                "SOURCE_REUSE_DEPENDENCY_INTEGRATION_REQUIRED: "
                + str(donor.get("repository"))
            )
        blobs: dict[str, bytes] = {}
        for item in donor.get("files", ()):
            relative = str(item.get("source_path") or "")
            if not relative:
                raise ValueError("SOURCE_REUSE_INVALID_PATH")
            blobs[relative] = Path(item["path"]).read_bytes()
        notice_path, notice_bytes = _verified_donor_license_notice(donor, blobs)
        prepared[notice_path] = (notice_bytes, str(donor["repository"]))
        adapted, _ = apply_deterministic_adapters(blobs, target_context)
        copied_for_donor = 0
        for relative, data in adapted.items():
            relative = str(relative).replace(chr(92), "/")
            if not relative.startswith(supported) or relative.endswith("/"):
                continue
            if not isinstance(data, (str, bytes)):
                raise ValueError("SOURCE_REUSE_NON_TEXT_OR_BYTES")
            raw = data.encode("utf-8") if isinstance(data, str) else data
            if relative in prepared and prepared[relative][0] != raw:
                raise RuntimeError("SOURCE_REUSE_ARTIFACT_COLLISION: " + relative)
            prepared[relative] = (raw, str(donor["repository"]))
            copied_for_donor += 1
        if not copied_for_donor:
            raise RuntimeError(
                "SOURCE_REUSE_NO_PRODUCTION_FILES: " + str(donor.get("repository"))
            )
        attribution.append({
            "repository": donor["repository"],
            "commit_sha": donor["commit_sha"],
            "license_id": donor["license_id"],
            "capability": donor["capability"],
        })

    # Validate path ownership and collisions before the first file write.
    for relative, (raw, _owner) in prepared.items():
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("SOURCE_REUSE_PATH_ESCAPE: " + relative) from exc
        if target.exists() and (not target.is_file() or target.read_bytes() != raw):
            raise RuntimeError("SOURCE_REUSE_EXISTING_FILE_CONFLICT: " + relative)

    for relative, (raw, _owner) in sorted(prepared.items()):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(raw)
    provenance_file = root / ".minecraft_ai" / "reuse" / "source_provenance.json"
    provenance_file.parent.mkdir(parents=True, exist_ok=True)
    provenance_file.write_text(
        json.dumps(attribution, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "mmm/authored-source-install-v1",
        "donor_count": len(attribution),
        "files": sorted(prepared),
        "provenance": str(provenance_file),
    }


__all__ = [
    "authored_capability_graph",
    "resolve_authored_source_reuse",
    "verified_reuse_context",
    "materialize_verified_authored_sources",
]
