from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected exactly one match, got {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# source_transplant: one license authority, O(1) queue, transient failures retry.
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    "from collections.abc import Mapping, Sequence\n",
    "from collections import deque\nfrom collections.abc import Mapping, Sequence\n",
)
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    'from .repository_artifact_index import RepositoryArtifactIndex\n\n_PERMISSIVE = frozenset({\n    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Zlib",\n    "Unlicense", "CC0-1.0",\n})\n',
    "from .repository_artifact_index import RepositoryArtifactIndex\nfrom .reuse_license import is_reusable_source_license\n",
)
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    '    queue: list[tuple[str, str]] = [(root_sha, "")]\n',
    '    queue: deque[tuple[str, str]] = deque([(root_sha, "")])\n',
)
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    "        tree_sha, prefix = queue.pop(0)\n",
    "        tree_sha, prefix = queue.popleft()\n",
)
source_path = Path("minecraft_mod_ai/source_transplant.py")
source_text = source_path.read_text(encoding="utf-8")
occurrences = source_text.count("license_id not in _PERMISSIVE")
if occurrences != 2:
    raise SystemExit(
        f"source_transplant.py: expected 2 legacy license checks, got {occurrences}"
    )
source_path.write_text(
    source_text.replace(
        "license_id not in _PERMISSIVE", "not is_reusable_source_license(license_id)"
    ),
    encoding="utf-8",
)
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    """    except Exception:\n        return None\n    finally:\n        with _SNAPSHOT_LOCK:\n            _SNAPSHOT_CACHE[repository] = snapshot\n            pending = _SNAPSHOT_INFLIGHT.pop(repository, None)\n            if pending is not None:\n                pending.set()\n""",
    """    except SourceTransplantError:\n        return None\n    finally:\n        with _SNAPSHOT_LOCK:\n            if snapshot is not None:\n                _SNAPSHOT_CACHE[repository] = snapshot\n            else:\n                _SNAPSHOT_CACHE.pop(repository, None)\n            pending = _SNAPSHOT_INFLIGHT.pop(repository, None)\n            if pending is not None:\n                pending.set()\n""",
)
replace_once(
    "minecraft_mod_ai/source_transplant.py",
    """def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:\n    response = client.get(url, params=params)\n    response.raise_for_status()\n    limit = _response_byte_budget()\n    if len(response.content) > limit:\n        raise SourceTransplantError(\n            f\"GitHub response exceeded configured source-transplant response budget ({limit} bytes).\"\n        )\n    return response.json()\n""",
    """def _github_json(client: httpx.Client, url: str, *, params: Mapping[str, str] | None = None) -> Any:\n    try:\n        response = client.get(url, params=params)\n        response.raise_for_status()\n    except httpx.HTTPError as exc:\n        raise SourceTransplantError(f\"GitHub donor request failed: {url}\") from exc\n    limit = _response_byte_budget()\n    if len(response.content) > limit:\n        raise SourceTransplantError(\n            f\"GitHub response exceeded configured source-transplant response budget ({limit} bytes).\"\n        )\n    try:\n        return response.json()\n    except ValueError as exc:\n        raise SourceTransplantError(\n            f\"GitHub donor response was not valid JSON: {url}\"\n        ) from exc\n""",
)

# proof executor: no mock-source proof, explicit target failure, no swallowed checker bugs.
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    "from .source_transplant import DonorSlice, SourceTransplantError\n\n\n@dataclass",
    'from .source_transplant import DonorSlice, SourceTransplantError\n\n\nclass ReuseTargetWorkspaceError(RuntimeError):\n    """Target workspace could not be copied into the isolated proof sandbox."""\n\n\n@dataclass',
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    "    if materialization_failed and not callable(compile_checker):\n",
    "    if materialization_failed:\n",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    '\n    if not in_memory_files and callable(compile_checker):\n        for donor_file in donor_slice.files:\n            in_memory_files[donor_file.path] = f"// Test Mock {donor_file.path}\\n"\n',
    "\n",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    """            try:\n                shutil.copytree(\n                    ws_path,\n                    sandbox_path,\n                    ignore=ignore_patterns,\n                    dirs_exist_ok=True,\n                )\n            except (OSError, shutil.Error):\n                pass\n""",
    """            try:\n                shutil.copytree(\n                    ws_path,\n                    sandbox_path,\n                    ignore=ignore_patterns,\n                    dirs_exist_ok=True,\n                )\n            except (OSError, shutil.Error) as exc:\n                raise ReuseTargetWorkspaceError(\n                    \"Failed to copy target workspace into reuse proof sandbox.\"\n                ) from exc\n""",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    """        if callable(compile_checker):\n            try:\n                check_result = compile_checker(adapted_files, target_context)\n                if isinstance(check_result, Mapping):\n                    compile_passed = bool(check_result.get(\"compile_passed\"))\n                    tests_passed = bool(check_result.get(\"tests_passed\"))\n                    tests_executed = int(check_result.get(\"tests_executed\", 0))\n                    tests_passed_count = int(\n                        check_result.get(\"tests_passed_count\", 0)\n                    )\n                    executed_test_ids = tuple(\n                        check_result.get(\"executed_test_ids\") or ()\n                    )\n                    individual_results = dict(\n                        check_result.get(\"individual_test_results\") or {}\n                    )\n                    unresolved_symbols.extend(\n                        check_result.get(\"unresolved_symbols\") or []\n                    )\n                    missing_resources.extend(\n                        check_result.get(\"missing_resources\") or []\n                    )\n                else:\n                    compile_passed = bool(check_result)\n                    tests_passed = False\n            except Exception:\n                compile_passed = False\n                tests_passed = False\n""",
    """        if callable(compile_checker):\n            check_result = compile_checker(adapted_files, target_context)\n            if isinstance(check_result, Mapping):\n                compile_passed = bool(check_result.get(\"compile_passed\"))\n                tests_passed = bool(check_result.get(\"tests_passed\"))\n                tests_executed = int(check_result.get(\"tests_executed\", 0))\n                tests_passed_count = int(check_result.get(\"tests_passed_count\", 0))\n                executed_test_ids = tuple(check_result.get(\"executed_test_ids\") or ())\n                individual_results = dict(\n                    check_result.get(\"individual_test_results\") or {}\n                )\n                unresolved_symbols.extend(check_result.get(\"unresolved_symbols\") or [])\n                missing_resources.extend(check_result.get(\"missing_resources\") or [])\n            else:\n                compile_passed = bool(check_result)\n                tests_passed = False\n""",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    """            if callable(compile_checker):\n                try:\n                    comp_result = compile_checker(comp_files, target_context)\n                    if isinstance(comp_result, Mapping):\n                        comp_passed = bool(comp_result.get(\"compile_passed\"))\n                    else:\n                        comp_passed = bool(comp_result)\n                except Exception:\n                    comp_passed = False\n""",
    """            if callable(compile_checker):\n                comp_result = compile_checker(comp_files, target_context)\n                if isinstance(comp_result, Mapping):\n                    comp_passed = bool(comp_result.get(\"compile_passed\"))\n                else:\n                    comp_passed = bool(comp_result)\n""",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    "    receipts: list[ReuseProofReceipt] = []\n    partial_candidate: DonorSlice | None = None\n",
    "    receipts: list[ReuseProofReceipt] = []\n    best_partial: tuple[tuple[int, int, int, int, int], DonorSlice] | None = None\n",
)
replace_once(
    "minecraft_mod_ai/reuse_proof_executor.py",
    """        if (\n            receipt_level == ProofLevel.PARTIAL_REUSE\n            and partial_candidate is None\n        ):\n            partial_candidate = candidate\n\n    if partial_candidate is not None:\n        return partial_candidate, tuple(receipts)\n""",
    """        if receipt_level == ProofLevel.PARTIAL_REUSE:\n            partial_score = (\n                len(receipt.verified_artifacts),\n                len(receipt.verified_symbols),\n                -len(receipt.residual_artifacts),\n                -len(receipt.unresolved_symbols),\n                -len(receipt.missing_resources),\n            )\n            if best_partial is None or partial_score > best_partial[0]:\n                best_partial = (partial_score, candidate)\n\n    if best_partial is not None:\n        return best_partial[1], tuple(receipts)\n""",
)

# planner: exact immutable donor identity, never a repository prefix.
replace_once(
    "minecraft_mod_ai/reuse_planner.py",
    "                winning_receipt = next((r for r in receipts if r.candidate_id.startswith(donor.repository)), receipts[-1] if receipts else None)\n",
    '                donor_identity = f"{donor.repository}@{donor.commit_sha}"\n                winning_receipt = next(\n                    (r for r in receipts if r.candidate_id == donor_identity),\n                    None,\n                )\n',
)

# final assembly: replay the exact configuration that executable proof verified.
replace_once(
    "minecraft_mod_ai/final_project_assembler.py",
    """            build_model.add_dependency(\n                str(resolved_coordinate),\n                \"modImplementation\" if self.target_loader == \"fabric\" else \"implementation\",\n                sha256=str(_receipt_value(dependency, \"artifact_hash\", \"\")),\n                requirement_ids=requirement_ids,\n            )\n""",
    """            gradle_configuration = str(\n                _receipt_value(dependency, \"gradle_configuration\", \"\")\n            ).strip()\n            if not gradle_configuration:\n                errors.append(\"RESOLVED_BUILD_DEPENDENCY_CONFIGURATION_MISSING\")\n                return\n            build_model.add_dependency(\n                str(resolved_coordinate),\n                gradle_configuration,\n                sha256=str(_receipt_value(dependency, \"artifact_hash\", \"\")),\n                requirement_ids=requirement_ids,\n            )\n""",
)

Path("tests/test_worker6_reuse_hardening.py").write_text(
    '''from __future__ import annotations\n\nimport hashlib\n\nimport pytest\n\nimport minecraft_mod_ai.reuse_proof_executor as reuse_proof\nimport minecraft_mod_ai.source_transplant as source_transplant\nfrom minecraft_mod_ai.proof_level import ProofLevel\nfrom minecraft_mod_ai.source_transplant import DonorFile, DonorSlice, SourceTransplantError\n\n\ndef _donor(repository: str = "example/worker6") -> DonorSlice:\n    payload = b"package donor; public class BossEntity {}\\n"\n    return DonorSlice(\n        capability="boss.entity", repository=repository, commit_sha="a" * 40,\n        license_id="MIT", source_url=f"https://github.com/{repository}",\n        target_compatibility="metadata_exact",\n        files=(DonorFile(path="src/main/java/donor/BossEntity.java", blob_sha="b" * 40,\n            sha256="sha256:" + hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),\n            symbols=("BossEntity",)),),\n        seed_files=("src/main/java/donor/BossEntity.java",), source_symbols=("BossEntity",),\n        required_dependencies=(), donor_tests=(), confidence=0.9, closure_complete=True,\n    )\n\n\ndef _partial(candidate: DonorSlice, verified: int, residual: int) -> reuse_proof.ReuseProofReceipt:\n    return reuse_proof.ReuseProofReceipt(\n        candidate_id=f"{candidate.repository}@{candidate.commit_sha}", capability=candidate.capability,\n        commit_sha=candidate.commit_sha, closure_hash="sha256:test",\n        proof_level=ProofLevel.PARTIAL_REUSE.value, compile_passed=False, tests_passed=False,\n        unresolved_symbols=(), missing_resources=(), adaptations_applied=(),\n        verified_capabilities=(), residual_capabilities=(candidate.capability,),\n        verified_artifacts=tuple(f"verified-{i}.java" for i in range(verified)),\n        residual_artifacts=tuple(f"residual-{i}.java" for i in range(residual)),\n        verified_symbols=tuple(f"Verified{i}" for i in range(verified)),\n    )\n\n\ndef test_materialization_failure_never_uses_mock_source_for_checker(monkeypatch, tmp_path):\n    checker_called = False\n    def fail_materialize(*args, **kwargs):\n        raise SourceTransplantError("donor unavailable")\n    def checker(*args, **kwargs):\n        nonlocal checker_called\n        checker_called = True\n        return True\n    monkeypatch.setattr(source_transplant, "materialize_pinned_donor", fail_materialize)\n    receipt = reuse_proof.execute_reuse_proof(\n        _donor(), target_workspace=tmp_path, target_context={}, compile_checker=checker\n    )\n    assert checker_called is False\n    assert receipt.compile_passed is False\n    assert not ProofLevel.from_value(receipt.proof_level).allows_reuse()\n\n\ndef test_compile_checker_programming_error_propagates(monkeypatch, tmp_path):\n    payload = b"package donor; public class BossEntity {}\\n"\n    monkeypatch.setattr(source_transplant, "materialize_pinned_donor",\n        lambda *args, **kwargs: {"src/main/java/donor/BossEntity.java": payload})\n    def explode(*args, **kwargs):\n        raise RuntimeError("checker programming error")\n    with pytest.raises(RuntimeError, match="checker programming error"):\n        reuse_proof.execute_reuse_proof(\n            _donor(), target_workspace=tmp_path, target_context={}, compile_checker=explode\n        )\n\n\ndef test_fallback_selects_strongest_partial_receipt(monkeypatch, tmp_path):\n    weak = _donor("example/weak")\n    strong = _donor("example/strong")\n    receipts = iter((_partial(weak, 1, 4), _partial(strong, 3, 1)))\n    monkeypatch.setattr(reuse_proof, "execute_reuse_proof",\n        lambda *args, **kwargs: next(receipts))\n    selected, all_receipts = reuse_proof.execute_candidate_fallback_loop(\n        (weak, strong), "boss.entity", target_workspace=tmp_path, target_context={}\n    )\n    assert selected is strong\n    assert len(all_receipts) == 2\n\n\ndef test_transient_repository_snapshot_failure_is_not_negative_cached():\n    source_transplant._SNAPSHOT_CACHE.clear()\n    source_transplant._SNAPSHOT_INFLIGHT.clear()\n    calls = 0\n    class Discovery:\n        github_token = ""\n        def inspect_github_repository(self, repository):\n            nonlocal calls\n            calls += 1\n            raise SourceTransplantError("transient")\n    discovery = Discovery()\n    assert source_transplant._repository_snapshot("example/transient", discovery) is None\n    assert source_transplant._repository_snapshot("example/transient", discovery) is None\n    assert calls == 2\n\n\ndef test_unexpected_snapshot_programming_error_is_not_hidden():\n    source_transplant._SNAPSHOT_CACHE.clear()\n    source_transplant._SNAPSHOT_INFLIGHT.clear()\n    class Discovery:\n        github_token = ""\n        def inspect_github_repository(self, repository):\n            raise RuntimeError("snapshot programming error")\n    with pytest.raises(RuntimeError, match="snapshot programming error"):\n        source_transplant._repository_snapshot("example/programming", Discovery())\n''',
    encoding="utf-8",
)
