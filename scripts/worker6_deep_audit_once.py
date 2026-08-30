from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, got {count}: {old[:100]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


source = "minecraft_mod_ai/source_transplant.py"
replace_once(source, "import base64\n", "import base64\nimport binascii\n")
replace_once(
    source,
    "from collections import deque\n",
    "from collections import OrderedDict, deque\n",
)
replace_once(
    source,
    '''def _tree_request_budget() -> int:\n    # Work budget only.  It never truncates a successfully enumerated tree.\n    return _env_int(\n        "MMM_SOURCE_TRANSPLANT_TREE_REQUEST_BUDGET",\n        2048,\n        minimum=8,\n        maximum=100_000,\n    )\n_SNAPSHOT_LOCK = Lock()\n_SNAPSHOT_CACHE: dict[str, Mapping[str, Any] | None] = {}\n_SNAPSHOT_INFLIGHT: dict[str, Event] = {}\n_BLOB_LOCK = Lock()\n_BLOB_CACHE: dict[tuple[str, str], bytes] = {}\n_BLOB_INFLIGHT: dict[tuple[str, str], Event] = {}\n''',
    '''def _tree_request_budget() -> int:\n    # Work budget only.  It never truncates a successfully enumerated tree.\n    return _env_int(\n        "MMM_SOURCE_TRANSPLANT_TREE_REQUEST_BUDGET",\n        2048,\n        minimum=8,\n        maximum=100_000,\n    )\n\n\ndef _snapshot_cache_entries() -> int:\n    return _env_int(\n        "MMM_SOURCE_TRANSPLANT_SNAPSHOT_CACHE_ENTRIES",\n        32,\n        minimum=1,\n        maximum=512,\n    )\n\n\ndef _blob_cache_byte_budget() -> int:\n    configured = _env_int(\n        "MMM_SOURCE_TRANSPLANT_BLOB_CACHE_BYTE_BUDGET",\n        256 * 1024 * 1024,\n        minimum=64 * 1024,\n        maximum=2 * 1024 * 1024 * 1024,\n    )\n    return max(configured, _single_blob_byte_budget())\n\n\n_SNAPSHOT_LOCK = Lock()\n_SNAPSHOT_CACHE: OrderedDict[str, Mapping[str, Any]] = OrderedDict()\n_SNAPSHOT_INFLIGHT: dict[str, Event] = {}\n_BLOB_LOCK = Lock()\n_BLOB_CACHE: OrderedDict[tuple[str, str], bytes] = OrderedDict()\n_BLOB_CACHE_BYTES = 0\n_BLOB_INFLIGHT: dict[tuple[str, str], Event] = {}\n''',
)
replace_once(
    source,
    '''    seen_trees: set[str] = set()\n    resolved: list[Mapping[str, Any]] = []\n    while queue:\n        if requests >= budget:\n            raise SourceTransplantError(\n                "Complete donor tree traversal exhausted the configured request budget."\n            )\n        tree_sha, prefix = queue.popleft()\n        if tree_sha in seen_trees:\n            continue\n        seen_trees.add(tree_sha)\n''',
    '''    seen_trees: set[tuple[str, str]] = set()\n    resolved: list[Mapping[str, Any]] = []\n    while queue:\n        if requests >= budget:\n            raise SourceTransplantError(\n                "Complete donor tree traversal exhausted the configured request budget."\n            )\n        tree_sha, prefix = queue.popleft()\n        tree_identity = (tree_sha, prefix)\n        if tree_identity in seen_trees:\n            continue\n        seen_trees.add(tree_identity)\n''',
)
replace_once(source, "    except Exception:\n        return None\n    finally:\n        client.close()\n\n\ndef materialize_source_slices", "    except SourceTransplantError:\n        return None\n    finally:\n        client.close()\n\n\ndef materialize_source_slices")
replace_once(
    source,
    '''    with _SNAPSHOT_LOCK:\n        if repository in _SNAPSHOT_CACHE:\n            return _SNAPSHOT_CACHE[repository]\n''',
    '''    with _SNAPSHOT_LOCK:\n        if repository in _SNAPSHOT_CACHE:\n            cached = _SNAPSHOT_CACHE[repository]\n            _SNAPSHOT_CACHE.move_to_end(repository)\n            return cached\n''',
)
replace_once(
    source,
    '''            if snapshot is not None:\n                _SNAPSHOT_CACHE[repository] = snapshot\n            else:\n                _SNAPSHOT_CACHE.pop(repository, None)\n''',
    '''            if snapshot is not None:\n                _SNAPSHOT_CACHE[repository] = snapshot\n                _SNAPSHOT_CACHE.move_to_end(repository)\n                while len(_SNAPSHOT_CACHE) > _snapshot_cache_entries():\n                    _SNAPSHOT_CACHE.popitem(last=False)\n            else:\n                _SNAPSHOT_CACHE.pop(repository, None)\n''',
)
replace_once(
    source,
    '''def _fetch_blob_bytes(client: httpx.Client, repository: str, blob_sha: str) -> bytes:\n    if not re.fullmatch(r"[0-9a-f]{40,64}", blob_sha):\n        raise SourceTransplantError("Donor blob is not immutable.")\n    key = (repository, blob_sha)\n    owner = False\n    with _BLOB_LOCK:\n        cached = _BLOB_CACHE.get(key)\n        if cached is not None:\n            return cached\n''',
    '''def _fetch_blob_bytes(client: httpx.Client, repository: str, blob_sha: str) -> bytes:\n    global _BLOB_CACHE_BYTES\n\n    if not re.fullmatch(r"[0-9a-f]{40,64}", blob_sha):\n        raise SourceTransplantError("Donor blob is not immutable.")\n    key = (repository, blob_sha)\n    owner = False\n    with _BLOB_LOCK:\n        cached = _BLOB_CACHE.get(key)\n        if cached is not None:\n            _BLOB_CACHE.move_to_end(key)\n            return cached\n''',
)
replace_once(
    source,
    '''        raw = base64.b64decode(str(value.get("content") or "").replace("\\n", ""), validate=True)\n        single_blob_budget = _single_blob_byte_budget()\n''',
    '''        try:\n            raw = base64.b64decode(\n                str(value.get("content") or "").replace("\\n", ""),\n                validate=True,\n            )\n        except (binascii.Error, ValueError) as exc:\n            raise SourceTransplantError("GitHub donor blob contained invalid base64.") from exc\n        single_blob_budget = _single_blob_byte_budget()\n''',
)
replace_once(
    source,
    '''        with _BLOB_LOCK:\n            while len(_BLOB_CACHE) >= 512:\n                _BLOB_CACHE.pop(next(iter(_BLOB_CACHE)))\n            _BLOB_CACHE[key] = raw\n        return raw\n''',
    '''        with _BLOB_LOCK:\n            existing = _BLOB_CACHE.pop(key, None)\n            if existing is not None:\n                _BLOB_CACHE_BYTES -= len(existing)\n            byte_budget = _blob_cache_byte_budget()\n            while _BLOB_CACHE and _BLOB_CACHE_BYTES + len(raw) > byte_budget:\n                _old_key, old_value = _BLOB_CACHE.popitem(last=False)\n                _BLOB_CACHE_BYTES -= len(old_value)\n            _BLOB_CACHE[key] = raw\n            _BLOB_CACHE_BYTES += len(raw)\n        return raw\n''',
)
replace_once(
    source,
    '''def materialize_pinned_donor(\n    donor_slice: DonorSlice,\n    discovery_client: Any = None,\n) -> dict[str, bytes]:\n''',
    '''def validate_donor_slice_manifest(donor_slice: DonorSlice) -> None:\n    """Validate immutable donor identity and every manifest path/hash before I/O."""\n\n    repository = str(donor_slice.repository or "").strip()\n    if repository.count("/") != 1 or any(\n        not part or part in {".", ".."} for part in repository.split("/")\n    ):\n        raise SourceTransplantError("Donor repository identity is invalid.")\n    if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_slice.commit_sha or "")):\n        raise SourceTransplantError("Donor commit is not an immutable full SHA.")\n    if not is_reusable_source_license(donor_slice.license_id):\n        raise SourceTransplantError("Donor source license is not admitted for reuse.")\n    if not donor_slice.files:\n        raise SourceTransplantError("Donor source-slice manifest is empty.")\n\n    seen_paths: set[str] = set()\n    for donor_file in donor_slice.files:\n        path = str(donor_file.path or "")\n        normalized = path.replace("\\\\", "/").strip()\n        parts = normalized.split("/")\n        if (\n            not normalized\n            or normalized != path\n            or normalized.startswith("/")\n            or re.match(r"^[A-Za-z]:", normalized)\n            or any(part in {"", ".", ".."} for part in parts)\n            or normalized in seen_paths\n        ):\n            raise SourceTransplantError("Donor manifest contains an unsafe or duplicate path.")\n        if not re.fullmatch(r"[0-9a-f]{40,64}", str(donor_file.blob_sha or "")):\n            raise SourceTransplantError("Donor manifest contains a non-immutable blob SHA.")\n        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(donor_file.sha256 or "").casefold()):\n            raise SourceTransplantError("Donor manifest contains an invalid SHA-256 binding.")\n        if donor_file.size_bytes < 0:\n            raise SourceTransplantError("Donor manifest contains a negative file size.")\n        seen_paths.add(normalized)\n\n\ndef materialize_pinned_donor(\n    donor_slice: DonorSlice,\n    discovery_client: Any = None,\n) -> dict[str, bytes]:\n''',
)
replace_once(
    source,
    '''    token = str(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()\n    client = getattr(discovery_client, "_client", None)\n''',
    '''    validate_donor_slice_manifest(donor_slice)\n    token = str(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()\n    client = getattr(discovery_client, "_client", None)\n''',
)
replace_once(
    source,
    '''            actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()\n            if actual_sha.casefold() != df.sha256.casefold():\n                raise SourceTransplantError(f"SHA-256 hash mismatch for {df.path}: expected {df.sha256}, got {actual_sha}")\n            materialized[df.path] = raw\n''',
    '''            actual_sha = "sha256:" + hashlib.sha256(raw).hexdigest()\n            if actual_sha.casefold() != df.sha256.casefold():\n                raise SourceTransplantError(\n                    f"SHA-256 hash mismatch for {df.path}: expected {df.sha256}, got {actual_sha}"\n                )\n            if len(raw) != df.size_bytes:\n                raise SourceTransplantError(\n                    f"Pinned donor size mismatch for {df.path}: expected {df.size_bytes}, got {len(raw)}"\n                )\n            materialized[df.path] = raw\n''',
)
replace_once(source, "        except Exception:\n            continue\n    return \"\\n\".join(chunks)\n", "        except SourceTransplantError:\n            continue\n    return \"\\n\".join(chunks)\n")
replace_once(source, '    "materialize_source_slices",\n', '    "materialize_source_slices",\n    "validate_donor_slice_manifest",\n')

# reuse artifacts: only donor/provenance failures are translated; exact proof symbols only.
artifacts = "minecraft_mod_ai/reuse_artifacts.py"
replace_once(
    artifacts,
    "from .source_transplant import DonorSlice, materialize_pinned_donor\n",
    "from .source_transplant import DonorSlice, SourceTransplantError, materialize_pinned_donor\n",
)
replace_once(
    artifacts,
    '''        protected_symbols = tuple(\n            _receipt_value(proof_receipt, "verified_symbols", ())\n            or donor.source_symbols\n        )\n''',
    '''        protected_symbols = tuple(\n            _receipt_value(proof_receipt, "verified_symbols", ()) or ()\n        )\n''',
)
replace_once(
    artifacts,
    '''            except Exception as exc:\n                raise BundleMaterializationError("Pinned donor materialization failed.") from exc\n''',
    '''            except (SourceTransplantError, KeyError, TypeError, ValueError) as exc:\n                raise BundleMaterializationError("Pinned donor materialization failed.") from exc\n''',
)
replace_once(
    artifacts,
    '''        return {\n            path: value\n            for raw_path, value in files.items()\n            if (path := _normalized_path(raw_path))\n        }\n''',
    '''        normalized_files: dict[str, str | bytes] = {}\n        for raw_path, value in files.items():\n            path = _normalized_path(raw_path)\n            if not path:\n                raise BundleMaterializationError("Materialized bundle contains an unsafe path.")\n            if path in normalized_files:\n                raise BundleMaterializationError("Materialized bundle contains duplicate paths.")\n            normalized_files[path] = value\n        return normalized_files\n''',
)

# proof executor: validate manifest before work, require exact dependency receipts, sandbox paths.
proof = "minecraft_mod_ai/reuse_proof_executor.py"
replace_once(proof, "import hashlib\n", "import hashlib\nimport json\n")
replace_once(
    proof,
    "from .source_transplant import DonorSlice, SourceTransplantError\n",
    "from .source_transplant import (\n    DonorSlice,\n    SourceTransplantError,\n    validate_donor_slice_manifest,\n)\n",
)
replace_once(
    proof,
    '''def _closure_sha256(donor_slice: DonorSlice) -> str:\n    combined = "".join(\n        f"{item.path}:{item.sha256}"\n        for item in sorted(donor_slice.files, key=lambda entry: entry.path)\n    )\n    return "sha256:" + hashlib.sha256(combined.encode("utf-8")).hexdigest()\n''',
    '''def _closure_sha256(donor_slice: DonorSlice) -> str:\n    payload = [\n        [item.path, item.blob_sha, item.sha256, item.size_bytes]\n        for item in sorted(donor_slice.files, key=lambda entry: entry.path)\n    ]\n    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))\n    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()\n''',
)
replace_once(
    proof,
    '''def scaffold_minimal_ephemeral_workspace(\n    sandbox_path: Path,\n    target_context: Mapping[str, Any],\n) -> None:\n''',
    '''def _sandbox_destination(root: Path, relative_path: Any) -> Path:\n    normalized = _safe_workspace_relative_path(relative_path)\n    if not normalized:\n        raise ReuseTargetWorkspaceError("Reuse proof artifact path is unsafe.")\n    root_resolved = root.resolve()\n    destination = root.joinpath(*normalized.split("/"))\n    try:\n        destination.resolve(strict=False).relative_to(root_resolved)\n    except (OSError, RuntimeError, ValueError) as exc:\n        raise ReuseTargetWorkspaceError("Reuse proof artifact escaped its sandbox.") from exc\n    return destination\n\n\ndef scaffold_minimal_ephemeral_workspace(\n    sandbox_path: Path,\n    target_context: Mapping[str, Any],\n) -> None:\n''',
)
replace_once(
    proof,
    '''        repository = str(_dependency_receipt_value(receipt, "repository", "")).strip()\n        coordinate = str(\n            _dependency_receipt_value(receipt, "resolved_coordinate", "")\n        ).strip()\n        if not coordinate:\n            raise ValueError("Resolved dependency receipt has no coordinate.")\n        if repository:\n            model.add_repository(repository)\n        model.add_dependency(\n            coordinate,\n            str(\n                _dependency_receipt_value(\n                    receipt,\n                    "gradle_configuration",\n                    "modImplementation"\n                    if model.target.loader == "fabric"\n                    else "implementation",\n                )\n            ),\n            sha256=str(_dependency_receipt_value(receipt, "artifact_hash", "")),\n        )\n''',
    '''        repository = str(_dependency_receipt_value(receipt, "repository", "")).strip()\n        coordinate = str(\n            _dependency_receipt_value(receipt, "resolved_coordinate", "")\n        ).strip()\n        configuration = str(\n            _dependency_receipt_value(receipt, "gradle_configuration", "")\n        ).strip()\n        fingerprint = str(\n            _dependency_receipt_value(receipt, "resolution_fingerprint", "")\n        ).strip()\n        if not repository or not coordinate or not configuration:\n            raise ValueError("Resolved dependency receipt lacks authoritative Gradle fields.")\n        if not re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint):\n            raise ValueError("Resolved dependency receipt lacks an authoritative fingerprint.")\n        model.add_repository(repository)\n        model.add_dependency(\n            coordinate,\n            configuration,\n            sha256=str(_dependency_receipt_value(receipt, "artifact_hash", "")),\n        )\n''',
)
replace_once(
    proof,
    '''    candidate_id = f"{donor_slice.repository}@{donor_slice.commit_sha}"\n    closure_hash = _closure_sha256(donor_slice)\n    current_level = ProofLevel.DISCOVERED\n\n    if not is_reusable_source_license(donor_slice.license_id):\n''',
    '''    candidate_id = f"{donor_slice.repository}@{donor_slice.commit_sha}"\n    closure_hash = _closure_sha256(donor_slice)\n    current_level = ProofLevel.DISCOVERED\n\n    try:\n        validate_donor_slice_manifest(donor_slice)\n    except SourceTransplantError:\n        return _license_rejected_receipt(\n            donor_slice, candidate_id=candidate_id, closure_hash=closure_hash\n        )\n\n    if not is_reusable_source_license(donor_slice.license_id):\n''',
)
replace_once(
    proof,
    '''    if donor_slice.commit_sha:\n        valid, _ = validate_proof_transition(\n            current_level,\n            ProofLevel.PINNED,\n            receipt={"commit_sha": donor_slice.commit_sha},\n        )\n        if valid:\n            current_level = ProofLevel.PINNED\n''',
    '''    valid, _ = validate_proof_transition(\n        current_level,\n        ProofLevel.PINNED,\n        receipt={"commit_sha": donor_slice.commit_sha},\n    )\n    if not valid:\n        return ReuseProofReceipt(\n            candidate_id=candidate_id, capability=donor_slice.capability,\n            commit_sha=donor_slice.commit_sha, closure_hash=closure_hash,\n            proof_level=current_level.value, compile_passed=False, tests_passed=False,\n            unresolved_symbols=(), missing_resources=(), adaptations_applied=(),\n            verified_capabilities=(), residual_capabilities=(donor_slice.capability,),\n        )\n    current_level = ProofLevel.PINNED\n''',
)
replace_once(
    proof,
    '''        for rel_path, content in adapted_files.items():\n            dest = sandbox_path / rel_path\n            dest.parent.mkdir(parents=True, exist_ok=True)\n''',
    '''        for rel_path, content in adapted_files.items():\n            dest = _sandbox_destination(sandbox_path, rel_path)\n            dest.parent.mkdir(parents=True, exist_ok=True)\n''',
)
replace_once(
    proof,
    '''    unresolved_set = set(unresolved_symbols)\n    verified_art_list: list[str] = []\n''',
    '''    unresolved_set = set(unresolved_symbols)\n    donor_symbols_by_path = {\n        donor_file.path: set(donor_file.symbols)\n        for donor_file in donor_slice.files\n    }\n    verified_art_list: list[str] = []\n''',
)
replace_once(
    proof,
    '''                donor_match = next(\n                    (\n                        donor_file\n                        for donor_file in donor_slice.files\n                        if donor_file.path == path\n                    ),\n                    None,\n                )\n                donor_symbols = set(donor_match.symbols) if donor_match else set()\n''',
    '''                donor_symbols = donor_symbols_by_path.get(path, set())\n''',
)
replace_once(
    proof,
    '''            else:\n                try:\n                    with tempfile.TemporaryDirectory(\n                        prefix="mmm_subgraph_"\n                    ) as sub_tmp:\n                        sub_path = Path(sub_tmp)\n                        scaffold_minimal_ephemeral_workspace(\n                            sub_path,\n                            target_context=target_context,\n                        )\n                        for relative_path, content in comp_files.items():\n                            destination = sub_path / relative_path\n                            destination.parent.mkdir(parents=True, exist_ok=True)\n                            if isinstance(content, bytes):\n                                destination.write_bytes(content)\n                            else:\n                                destination.write_text(str(content), encoding="utf-8")\n                        _render_proof_build_model(\n                            sub_path,\n                            target_context,\n                            exact_dependency_receipts,\n                        )\n                        from .reuse_build_verifier import verify_scratch_workspace_build\n\n                        sub_receipt = verify_scratch_workspace_build(\n                            sub_path,\n                            run_tests=False,\n                        )\n                        comp_passed = sub_receipt.compile_passed\n                except Exception:\n                    comp_passed = False\n''',
    '''            else:\n                with tempfile.TemporaryDirectory(prefix="mmm_subgraph_") as sub_tmp:\n                    sub_path = Path(sub_tmp)\n                    scaffold_minimal_ephemeral_workspace(\n                        sub_path,\n                        target_context=target_context,\n                    )\n                    for relative_path, content in comp_files.items():\n                        destination = _sandbox_destination(sub_path, relative_path)\n                        destination.parent.mkdir(parents=True, exist_ok=True)\n                        if isinstance(content, bytes):\n                            destination.write_bytes(content)\n                        else:\n                            destination.write_text(str(content), encoding="utf-8")\n                    _render_proof_build_model(\n                        sub_path,\n                        target_context,\n                        exact_dependency_receipts,\n                    )\n                    from .reuse_build_verifier import verify_scratch_workspace_build\n\n                    sub_receipt = verify_scratch_workspace_build(\n                        sub_path,\n                        run_tests=False,\n                    )\n                    comp_passed = sub_receipt.compile_passed\n''',
)

# Planner helper makes exact proof binding explicit and unit-testable.
planner = "minecraft_mod_ai/reuse_planner.py"
replace_once(
    planner,
    '''                donor_identity = f"{donor.repository}@{donor.commit_sha}"\n                winning_receipt = next(\n                    (r for r in receipts if r.candidate_id == donor_identity),\n                    None,\n                )\n''',
    '''                donor_identity = f"{donor.repository}@{donor.commit_sha}"\n                matching_receipts = tuple(\n                    receipt for receipt in receipts\n                    if receipt.candidate_id == donor_identity\n                )\n                winning_receipt = (\n                    matching_receipts[0] if len(matching_receipts) == 1 else None\n                )\n''',
)

# Dependency resolution: precompute alias index once instead of rebuilding sets per lookup.
deps = "minecraft_mod_ai/dependency_resolver.py"
replace_once(
    deps,
    '''def _canonical_dependency_key(raw: str) -> str:\n    token = _normalized_dependency_token(raw)\n    if not token:\n        return ""\n    for key, entry in _CANONICAL_DEPENDENCY_REGISTRY.items():\n        aliases = {\n            _normalized_dependency_token(alias)\n            for alias in entry.get("aliases", ())\n        }\n        aliases.add(_normalized_dependency_token(key))\n        for artifact in entry.get("name_by_loader", {}).values():\n            aliases.add(_normalized_dependency_token(artifact))\n            aliases.add(\n                _normalized_dependency_token(f"{entry.get('group', '')}:{artifact}")\n            )\n        if token in aliases:\n            return key\n    return ""\n''',
    '''def _build_dependency_alias_index() -> dict[str, str]:\n    index: dict[str, str] = {}\n    for key, entry in _CANONICAL_DEPENDENCY_REGISTRY.items():\n        aliases = set(entry.get("aliases", ()))\n        aliases.add(key)\n        for artifact in entry.get("name_by_loader", {}).values():\n            aliases.add(str(artifact))\n            aliases.add(f"{entry.get('group', '')}:{artifact}")\n        for alias in aliases:\n            token = _normalized_dependency_token(alias)\n            if not token:\n                continue\n            previous = index.get(token)\n            if previous is not None and previous != key:\n                raise RuntimeError(\n                    f"Dependency alias collision: {alias!r} maps to {previous!r} and {key!r}"\n                )\n            index[token] = key\n    return index\n\n\n_CANONICAL_DEPENDENCY_ALIAS_INDEX = _build_dependency_alias_index()\n\n\ndef _canonical_dependency_key(raw: str) -> str:\n    token = _normalized_dependency_token(raw)\n    return _CANONICAL_DEPENDENCY_ALIAS_INDEX.get(token, "") if token else ""\n''',
)

# Extend Worker 6 regression suite.
test_path = Path("tests/test_worker6_reuse_hardening.py")
test_text = test_path.read_text(encoding="utf-8")
test_text += r'''


def test_invalid_commit_is_rejected_before_materialization(monkeypatch, tmp_path):
    donor = _donor()
    invalid = source_transplant.DonorSlice(
        **{**donor.__dict__, "commit_sha": "abc123"}
    )
    called = False
    def materialize(*args, **kwargs):
        nonlocal called
        called = True
        return {}
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", materialize)
    receipt = reuse_proof.execute_reuse_proof(
        invalid, target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: True
    )
    assert called is False
    assert receipt.compile_passed is False
    assert receipt.proof_level == ProofLevel.DISCOVERED.value


def test_unsafe_manifest_path_is_rejected_before_io(monkeypatch, tmp_path):
    donor = _donor()
    bad_file = source_transplant.DonorFile(
        path="../escape.java", blob_sha="b" * 40,
        sha256=donor.files[0].sha256, size_bytes=donor.files[0].size_bytes,
        symbols=("BossEntity",),
    )
    unsafe = source_transplant.DonorSlice(
        **{**donor.__dict__, "files": (bad_file,)}
    )
    called = False
    def materialize(*args, **kwargs):
        nonlocal called
        called = True
        return {}
    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", materialize)
    receipt = reuse_proof.execute_reuse_proof(
        unsafe, target_workspace=tmp_path, target_context={}, compile_checker=lambda *_: True
    )
    assert called is False
    assert receipt.compile_passed is False


def test_pinned_materialization_verifies_declared_size(monkeypatch):
    donor = _donor()
    wrong_size = source_transplant.DonorSlice(
        **{**donor.__dict__, "files": (
            source_transplant.DonorFile(
                path=donor.files[0].path, blob_sha=donor.files[0].blob_sha,
                sha256=donor.files[0].sha256, size_bytes=donor.files[0].size_bytes + 1,
                symbols=donor.files[0].symbols,
            ),
        )}
    )
    payload = b"package donor; public class BossEntity {}\n"
    monkeypatch.setattr(source_transplant, "_fetch_blob_bytes", lambda *args, **kwargs: payload)
    with pytest.raises(SourceTransplantError, match="size mismatch"):
        source_transplant.materialize_pinned_donor(
            wrong_size, discovery_client=type("D", (), {"_client": object()})()
        )


def test_reused_tree_sha_is_walked_for_each_prefix(monkeypatch):
    root = "1" * 40
    shared = "2" * 40
    commit = "a" * 40
    def fake_json(client, url, *, params=None):
        del client
        if "/git/commits/" in url:
            return {"tree": {"sha": root}}
        if url.endswith(root) and params == {"recursive": "1"}:
            return {"truncated": True, "tree": []}
        if url.endswith(root):
            return {"truncated": False, "tree": [
                {"path": "a", "sha": shared, "type": "tree"},
                {"path": "b", "sha": shared, "type": "tree"},
            ]}
        if url.endswith(shared):
            return {"truncated": False, "tree": [
                {"path": "Boss.java", "sha": "3" * 40, "type": "blob"}
            ]}
        raise AssertionError(url)
    monkeypatch.setattr(source_transplant, "_github_json", fake_json)
    entries = source_transplant._repository_tree_entries(object(), "example/repo", commit)
    assert {item["path"] for item in entries} == {"a/Boss.java", "b/Boss.java"}


def test_blob_cache_is_byte_bounded_lru(monkeypatch):
    source_transplant._BLOB_CACHE.clear()
    source_transplant._BLOB_CACHE_BYTES = 0
    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_SINGLE_BLOB_BYTE_BUDGET", str(64 * 1024))
    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_BLOB_CACHE_BYTE_BUDGET", str(128 * 1024))
    payloads = {
        "1" * 40: b"a" * 60_000,
        "2" * 40: b"b" * 60_000,
        "3" * 40: b"c" * 60_000,
    }
    import base64
    def fake_json(client, url, *, params=None):
        del client, params
        sha = url.rsplit("/", 1)[-1]
        return {"encoding": "base64", "content": base64.b64encode(payloads[sha]).decode()}
    monkeypatch.setattr(source_transplant, "_github_json", fake_json)
    for sha in payloads:
        source_transplant._fetch_blob_bytes(object(), "example/repo", sha)
    assert source_transplant._BLOB_CACHE_BYTES <= source_transplant._blob_cache_byte_budget()
    assert len(source_transplant._BLOB_CACHE) == 2
    assert ("example/repo", "1" * 40) not in source_transplant._BLOB_CACHE


def test_sandbox_destination_rejects_parent_escape(tmp_path):
    with pytest.raises(reuse_proof.ReuseTargetWorkspaceError):
        reuse_proof._sandbox_destination(tmp_path, "../escape.java")
'''
test_path.write_text(test_text, encoding="utf-8")
