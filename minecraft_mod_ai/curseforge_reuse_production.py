"""Executable, fail-closed CurseForge source reuse integrated at production time.

Only sources explicitly linked by a CurseForge catalogue record are examined.
The host performs a real isolated Gradle build using MMM's existing proof engine.
Never promotes model decisions or an injected proof receipt without verification.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import copy
import hashlib
import json
import re

from .curseforge_semantic_source_review import (
    SourceInspectionError, direct_source_identity, _safe_archive_files,
    _read_bounded, _license_spdx,
)
from .reuse_license import is_reusable_source_license

_MAX_TRANSPLANT_FILES = 28
_MAX_PROOF_CANDIDATES_PER_FEATURE = 2
_MAX_PROOF_CANDIDATES_PER_RUN = 8
_JAVA_PACKAGE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][\w.]*)\s*;")
_JAVA_IMPORT = re.compile(r"(?m)^\s*import\s+(?:static\s+)?([A-Za-z_][\w.]*)\s*;")
_JAVA_TYPE = re.compile(r"\b(?:class|interface|record|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
_JAVA_TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_JAVA_METHOD = re.compile(r"\b(?:public|protected|private)\s+(?:static\s+)?[\w<>\[\], ?]+\s+(\w+)\s*\(")


def _source_url_matches(source_url: str, repository: str) -> tuple[str, str]:
    host, owner, name = direct_source_identity(source_url)
    if host not in {"github.com", "gitlab.com"} or owner + "/" + name != repository:
        raise SourceInspectionError("SOURCE_CATALOGUE_ORIGIN_MISMATCH")
    return owner, name


def fetch_pinned_archive(source_url: str, commit_sha: str, *, repository: str = "") -> dict[str, bytes]:
    """Download the *exact* commit, never mutable HEAD or an arbitrary URL.

    The compressed response is bounded and ZIP entries are path/size checked.
    No subprocess, GitHub search, or repository scripts are run.
    """
    from urllib.parse import quote
    import httpx
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit_sha or "")):
        raise SourceInspectionError("SOURCE_COMMIT_NOT_PINNED")
    if not repository:
        raise SourceInspectionError("SOURCE_REPOSITORY_NOT_BOUND")
    owner, name = _source_url_matches(source_url, repository)
    host, _, _ = direct_source_identity(source_url)
    with httpx.Client() as client:
        if host == "github.com":
            url = (f"https://codeload.github.com/{quote(owner, safe='')}/"
                   f"{quote(name, safe='')}/zip/{commit_sha}")
        else:
            project = quote(owner + "/" + name, safe="")
            url = (f"https://gitlab.com/api/v4/projects/{project}/repository/"
                   f"archive.zip?sha={commit_sha}")
        payload = _read_bounded(client, url, accept="application/zip")
    return _safe_archive_files(payload)


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def _canonical_source_path(path: str) -> bool:
    """Scope proof/install to MMM-controlled compile resources, not build scripts."""
    return (path.startswith(("src/main/java/", "src/client/java/"))
            and path.endswith(".java") and not any(x in {"", ".", ".."}
            for x in path.split("/")))


def _choose_closed_java_slice(files: Mapping[str, bytes], selected: str) -> list[str]:
    """Follow local Java types/imports, fail closed for unresolved or oversized closure.

    This is conservative: only declarations within the same source set, with
    canonical paths. External API symbols are validated by Gradle, not guessed.
    """
    sources = {p: raw.decode("utf-8", errors="replace") for p, raw in files.items()
               if _canonical_source_path(p)}
    if selected not in sources:
        raise SourceInspectionError("SOURCE_SELECTION_FILE_NOT_FOUND")
    packages: dict[str, str] = {}
    types_by_fqn: dict[str, str] = {}
    for p, body in sources.items():
        match = _JAVA_PACKAGE.search(body)
        pkg = match.group(1) if match else ""
        packages[p] = pkg
        for kind in _JAVA_TYPE.findall(body[:32000]):
            types_by_fqn[(pkg + "." if pkg else "") + kind] = p
    queue = deque([selected])
    collected: list[str] = []
    seen: set[str] = set()
    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)
        collected.append(path)
        if len(collected) > _MAX_TRANSPLANT_FILES:
            raise SourceInspectionError("SOURCE_SLICE_DEPENDENCY_CLOSURE_TOO_LARGE")
        body = sources[path]
        pkg = packages[path]
        tokens = set(_JAVA_TOKEN.findall(body))
        # Imports and referenced same-package class declarations are both included.
        imported = set(_JAVA_IMPORT.findall(body))
        for fqn, provider in types_by_fqn.items():
            if provider == path:
                continue
            name = fqn.rsplit(".", 1)[-1]
            if fqn in imported or (packages[provider] == pkg and name in tokens):
                if provider not in seen:
                    queue.append(provider)
    return sorted(collected)


def _license_path(files: Mapping[str, bytes], claimed_spdx: str) -> str:
    license_spdx, _ = _license_spdx(files)
    if not license_spdx or not is_reusable_source_license(license_spdx):
        raise SourceInspectionError("SOURCE_LICENSE_NOT_APPROVED")
    if license_spdx != claimed_spdx:
        raise SourceInspectionError("SOURCE_LICENSE_CHANGED_SINCE_REVIEW")
    accepted = {"LICENSE", "LICENSE.TXT", "LICENSE.MD", "COPYING", "COPYING.TXT"}
    licenses = [p for p in files if p.upper() in accepted]
    if not licenses:
        raise SourceInspectionError("SOURCE_LICENSE_NOTICE_MISSING")
    return sorted(licenses)[0]


def donor_from_reference(reference: Mapping[str, Any], files: Mapping[str, bytes]) -> dict[str, Any]:
    """Return a real MMM DonorSlice manifest; never trust model-generated hashes."""
    capability = str(reference.get("capability") or "").strip()
    selected = str(reference.get("selected_file") or "")
    symbol = str(reference.get("selected_symbol") or "")
    url = str(reference.get("source_url") or "")
    sha = str(reference.get("pinned_sha") or "")
    license_id = str(reference.get("license_spdx") or "")
    from .curseforge_semantic_source_review import direct_source_identity
    host, owner, name = direct_source_identity(url)
    if host not in {"github.com", "gitlab.com"}:
        raise SourceInspectionError("SOURCE_HOST_UNSUPPORTED_FOR_PROOF")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SourceInspectionError("SOURCE_COMMIT_NOT_PINNED")
    if not capability or not _canonical_source_path(selected) or selected not in files:
        raise SourceInspectionError("SOURCE_REFERENCE_NOT_GROUNDED")
    selected_raw = files[selected]
    digest = "sha256:" + hashlib.sha256(selected_raw).hexdigest()
    if digest != reference.get("file_sha256"):
        raise SourceInspectionError("SOURCE_REFERENCE_HASH_CHANGED")
    selected_source = selected_raw.decode("utf-8", errors="replace")
    if symbol not in set(_JAVA_TYPE.findall(selected_source) + _JAVA_METHOD.findall(selected_source)):
        raise SourceInspectionError("SOURCE_SELECTED_TYPE_NOT_DECLARED")
    license_path = _license_path(files, license_id)
    paths = _choose_closed_java_slice(files, selected)
    # Metadata is evidence for dependency resolution, NOT source to copy over
    # the target's own Fabric entrypoint/build files.
    from .dependency_resolver import parse_donor_build_metadata
    metadata = {
        path: body for path, body in files.items()
        if path in {"build.gradle", "build.gradle.kts", "gradle.properties",
                    "src/main/resources/fabric.mod.json"}
        or path.endswith("/gradle/libs.versions.toml")
    }
    required_dependencies = list(parse_donor_build_metadata(metadata))
    java_sources = "\n".join(files[p].decode("utf-8", errors="replace") for p in paths)
    if ("org.spongepowered.asm.mixin" in java_sources
            or "@Mixin(" in java_sources
            or ".accesswidener" in java_sources):
        raise SourceInspectionError("SOURCE_MIXIN_REQUIRES_EXPLICIT_INTEGRATION")

    # A Java-only manifest silently drops models, textures and data. Include
    # conventional resources, excluding donor mod entrypoints/build settings,
    # and refuse an oversized/complex closure rather than claiming success.
    resources = [
        p for p in sorted(files)
        if (p.startswith(("src/main/resources/", "src/client/resources/"))
            and ("/assets/" in p or "/data/" in p)
            and not p.endswith((".mixins.json", ".accesswidener")))
    ]
    if len(resources) > 64 or sum(len(files[p]) for p in resources) > 8 * 1024 * 1024:
        raise SourceInspectionError("SOURCE_RESOURCE_CLOSURE_NEEDS_COMPONENT_SELECTION")
    paths.extend(resources)
    paths.append(license_path)
    paths.extend(p for p in sorted(files)
                 if p.upper() in {"NOTICE", "NOTICE.TXT", "NOTICE.MD", "COPYRIGHT"})
    paths = list(dict.fromkeys(paths))
    donor_files = []
    for path in paths:
        raw = files[path]
        donor_files.append({
            "path": path,
            "blob_sha": git_blob_sha1(raw),
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "symbols": (_JAVA_TYPE.findall(raw.decode("utf-8", errors="replace"))[:40]
                        if path.endswith(".java") else []),
        })
    score = reference.get("semantic_score")
    confidence = max(0.0, min(1.0, float(score) / 100.0)) if type(score) in (int, float) else 0.0
    return {
        "schema_version": "mmm/source-transplant-slice-v1",
        "capability": capability,
        "repository": owner + "/" + name,
        "commit_sha": sha,
        "license_id": license_id,
        "source_url": url,
        "target_compatibility": "adapt",  # Must PROVE selected Minecraft target.
        "files": donor_files,
        "seed_files": [selected],
        "source_symbols": [symbol],
        "required_dependencies": required_dependencies,  # Resolved by target build proof.
        "donor_tests": [],
        "confidence": confidence,
        "adaptation_cost": 1.0 - confidence,
        "closure_complete": True,  # Bounded source/resource closure; behavior proof still mandatory.
        "truncation_reason": "",
    }


def read_pinned_manifest(donor: Any) -> dict[str, bytes]:
    """Re-download and verify each declared source byte for build/install.

    No pre-existing model receipt authorizes unchecked paths or bytes.
    """
    from .source_transplant import validate_donor_slice_manifest, SourceTransplantError
    validate_donor_slice_manifest(donor)
    try:
        files = fetch_pinned_archive(donor.source_url, donor.commit_sha,
                                     repository=donor.repository)
        result: dict[str, bytes] = {}
        for entry in donor.files:
            raw = files.get(entry.path)
            if raw is None:
                raise SourceInspectionError("SOURCE_PINNED_FILE_MISSING")
            if (len(raw) != entry.size_bytes
                    or "sha256:" + hashlib.sha256(raw).hexdigest() != entry.sha256
                    or git_blob_sha1(raw) != entry.blob_sha):
                raise SourceInspectionError("SOURCE_PINNED_FILE_HASH_MISMATCH")
            result[entry.path] = raw
        return result
    except SourceInspectionError as exc:
        raise SourceTransplantError(str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise SourceTransplantError("SOURCE_PINNED_FETCH_FAILED: " + type(exc).__name__) from exc
    except Exception as exc:
        # Network or archive processing errors must remain fail-closed, not abort
        # the remaining candidate/fresh implementation pipeline.
        import httpx
        if isinstance(exc, httpx.HTTPError):
            raise SourceTransplantError("SOURCE_PINNED_HTTP_FAILED") from exc
        raise


def stage_verified_donors(project_root: str | Path, reuse_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Stage *only* full-proof donors, with attested bytes, for the existing installer."""
    from .source_transplant import validated_reuse_donor
    root = Path(project_root).resolve()
    staging = root / ".minecraft_ai" / "reuse" / "staged_verified"
    donors: list[dict[str, Any]] = []
    decisions = reuse_plan.get("capabilities", ())
    for row in decisions:
        if not isinstance(row, Mapping) or row.get("mode") != "source_transplant":
            continue
        donor = validated_reuse_donor(row)
        data = read_pinned_manifest(donor)
        owner = hashlib.sha256((donor.repository + "@" + donor.commit_sha).encode()).hexdigest()[:20]
        prepared = []
        for path, raw in data.items():
            dest = staging / owner / path
            root_resolved = staging.resolve()
            if not dest.resolve().is_relative_to(root_resolved):
                raise ValueError("SOURCE_STAGE_PATH_ESCAPE")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                raise ValueError("SOURCE_STAGE_SYMLINK_BLOCKED")
            if dest.exists() and dest.read_bytes() != raw:
                raise ValueError("SOURCE_STAGE_BYTE_COLLISION")
            dest.write_bytes(raw)
            prepared.append({"source_path": path, "path": str(dest),
                             "sha256": "sha256:" + hashlib.sha256(raw).hexdigest()})
        donors.append({"capability": donor.capability, "repository": donor.repository,
                       "commit_sha": donor.commit_sha, "license_id": donor.license_id,
                       "required_dependencies": list(donor.required_dependencies),
                       "files": prepared})
    return {"schema_version": "mmm/reuse-materialization-v1",
            "donors": donors, "count": len(donors)}


def reject_existing_workspace_collisions(donor: Any, raw_files: Mapping[str, bytes],
                                         target_context: Mapping[str, Any],
                                         project_root: str | Path) -> None:
    """A scratch compile must not overwrite a production scaffold to pass."""
    from .reuse_adapters import apply_deterministic_adapters
    existing = {name: raw_files[name] for name in (item.path for item in donor.files)}
    adapted, _ = apply_deterministic_adapters(existing, target_context)
    root = Path(project_root).resolve()
    for relative, value in adapted.items():
        if not relative.startswith(("src/main/java/", "src/client/java/",
                                    "src/main/resources/", "src/client/resources/")):
            continue
        destination = root / relative
        if not destination.resolve().is_relative_to(root):
            raise SourceInspectionError("SOURCE_PROOF_WORKSPACE_PATH_ESCAPE")
        if destination.is_symlink():
            raise SourceInspectionError("SOURCE_PROOF_WORKSPACE_SYMLINK")
        raw = value.encode("utf-8") if isinstance(value, str) else value
        if destination.exists() and (not destination.is_file() or destination.read_bytes() != raw):
            raise SourceInspectionError("SOURCE_PROOF_EXISTING_FILE_COLLISION: " + relative)


def certify_sources_at_production(reuse_plan: Mapping[str, Any], *,
                                  project_root: str | Path,
                                  target_context: Mapping[str, Any],
                                  proof_runner: Any = None,
                                  source_fetcher: Any = None) -> dict[str, Any]:
    """Build selected source in an isolated target and promote only authoritative receipts.

    Optional injected callbacks are for isolated unit tests. The production caller
    supplies neither and ALWAYS executes the real MMM Gradle/JUnit proof engine.
    """
    from .source_transplant import DonorSlice, validated_reuse_donor
    from .reuse_proof_executor import execute_reuse_proof
    from .proof_level import ProofLevel
    from .platform_catalog import adapter_for_target

    bound = dict(reuse_plan.get("bound_target") or {})
    if (bound.get("minecraft_version") != target_context.get("minecraft_version")
            or bound.get("loader") != target_context.get("loader")):
        raise ValueError("SOURCE_PROOF_TARGET_MISMATCH")
    adapter_for_target(bound["minecraft_version"], bound["loader"]).validate()
    plan = copy.deepcopy(dict(reuse_plan))
    references = list(plan.get("reference_sources") or ())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in references:
        if not isinstance(ref, Mapping) or ref.get("reference_only") is not True:
            continue
        if ref.get("proof_status") != "NOT_RUN":
            continue
        # A documentation reference is not a donor recommendation. The model
        # must ground functional reuse in actual code before any Gradle proof.
        scores = ref.get("semantic_scores") or {}
        if (ref.get("semantic_intent") not in {"adapt", "reuse"}
                or not isinstance(scores, Mapping)
                or int(scores.get("functional_fit") or 0) < 3
                or int(scores.get("completeness") or 0) < 2):
            continue
        grouped.setdefault(str(ref.get("capability") or ""), []).append(dict(ref))
    for group in grouped.values():
        group.sort(key=lambda r: -float(r.get("semantic_score") or 0))

    proof_receipts = list(plan.get("proof_receipts") or ())
    attempts = 0
    for decision in plan.get("capabilities", ()):
        if not isinstance(decision, dict) or decision.get("mode") != "fresh":
            continue
        capability = decision.get("capability")
        feature_attempts = 0
        for ref in grouped.get(capability, ()):
            if attempts >= _MAX_PROOF_CANDIDATES_PER_RUN or feature_attempts >= _MAX_PROOF_CANDIDATES_PER_FEATURE:
                break
            # A semantic model result is not permission to skip source + license checks.
            if ref.get("semantic_score", 0) < 1:
                continue
            attempts += 1
            feature_attempts += 1
            try:
                url = str(ref.get("source_url") or "")
                host, owner, repo = direct_source_identity(url)
                if host not in {"github.com", "gitlab.com"}:
                    raise SourceInspectionError("SOURCE_PROOF_HOST_UNSUPPORTED")
                f = source_fetcher or fetch_pinned_archive
                files = f(url, str(ref.get("pinned_sha") or ""), repository=owner + "/" + repo)
                donor_dict = donor_from_reference(ref, files)
                donor = DonorSlice.from_dict(donor_dict)
                reject_existing_workspace_collisions(donor, files, target_context, project_root)
                runner = proof_runner or execute_reuse_proof
                proof = runner(donor, target_workspace=project_root,
                               target_context=target_context, run_tests=True)
                receipt = proof.to_dict() if hasattr(proof, "to_dict") else dict(proof)
                # Only proof from the actual target build may receive authority.
                if (receipt.get("authoritative_compile") is not True
                        or receipt.get("compile_passed") is not True
                        or not ProofLevel.from_value(receipt.get("proof_level")).is_verified()):
                    proof_receipts.append({"capability": capability, "candidate_id":
                                           donor.repository + "@" + donor.commit_sha,
                                           "status": "proof_result", "receipt": receipt})
                    continue
                from .curseforge_source_attestation import attest_compiled_donor_files
                receipt["verified_file_attestations"] = attest_compiled_donor_files(
                    donor, files, receipt, target_context,
                )
                # Compilation proves syntactic/API compatibility, not gameplay.
                # Only an implementation-bound host acceptance test can promote
                # the donor to verified feature reuse. Otherwise preserve it
                # as a diagnosed reference while the feature remains fresh.
                if (receipt.get("tests_passed") is not True
                        or not receipt.get("matched_capability_tests")):
                    proof_receipts.append({
                        "capability": capability,
                        "candidate_id": donor.repository + "@" + donor.commit_sha,
                        "status": "compile_only_unverified_behavior",
                        "receipt": receipt,
                    })
                    continue
                candidate = {**decision, "mode": "source_transplant",
                             "source_id": "host-donor:" + donor.repository + "@" + donor.commit_sha,
                             "donor": donor_dict, "proof_receipt": receipt}
                validated_reuse_donor(candidate)
                proof_receipts.append({"capability": capability, "candidate_id":
                                       donor.repository + "@" + donor.commit_sha,
                                       "status": "proof_result", "receipt": receipt})
                decision.clear()
                decision.update(candidate)
                break
            except Exception as exc:
                proof_receipts.append({"capability": capability,
                                      "candidate_id": str(ref.get("source_id") or ""),
                                      "status": "proof_error", "error": type(exc).__name__,
                                      "failure_message": str(exc)[:400] if isinstance(exc, (SourceInspectionError, ValueError)) else ""})
        # Features without compiled donors remain fresh, with their source hints intact.
    plan["proof_receipts"] = proof_receipts
    plan["proof_attempt_count"] = attempts
    print("SOURCE_REUSE_PROOF_SUMMARY: " + json.dumps({
        "attempted": attempts,
        "compiled_donors": sum(row.get("mode") == "source_transplant"
                               for row in plan.get("capabilities", ()) if isinstance(row, Mapping)),
        "compiled_unverified_behavior": sum(
            row.get("status") == "compile_only_unverified_behavior"
            for row in proof_receipts if isinstance(row, Mapping)),
        "failed": sum(row.get("status") == "proof_error" or
                      (row.get("status") == "proof_result" and
                       not (row.get("receipt") or {}).get("compile_passed"))
                      for row in proof_receipts if isinstance(row, Mapping)),
    }, ensure_ascii=False), flush=True)
    return plan


__all__ = ["certify_sources_at_production", "stage_verified_donors",
           "donor_from_reference", "read_pinned_manifest", "fetch_pinned_archive",
           "git_blob_sha1"]
