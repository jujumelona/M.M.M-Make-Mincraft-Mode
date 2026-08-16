from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

from minecraft_mod_ai import reuse_asset_upgrade_contract as contract
from minecraft_mod_ai import source_transplant as transplant


class _ClosableClient:
    def close(self) -> None:
        pass


def test_joint_reuse_optimizer_is_the_host_optimizer_hook(monkeypatch) -> None:
    original_calls: list[str] = []

    def original(prompt, **kwargs):
        original_calls.append(prompt)
        return SimpleNamespace(selected="legacy")

    resolver = SimpleNamespace(_optimize=original, SpecValidationError=RuntimeError)
    base = SimpleNamespace(selected="joint")
    plan = SimpleNamespace(to_dict=lambda: {"target": {"minecraft_version": "1.21.1", "loader": "fabric"}})
    joint = SimpleNamespace(base_optimization=base, selected_plan=plan)
    monkeypatch.setattr(contract, "optimize_platform_and_reuse", lambda *args, **kwargs: joint)
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "auto")

    contract._install_joint_platform_optimizer(resolver)
    result = resolver._optimize(
        "trade mod",
        design={},
        module_kinds=("custom_java",),
        loader_constraint=None,
        version_constraint=None,
        target_research_fn=None,
    )

    assert result is base
    assert original_calls == []
    assert result._mmm_reuse_plan["target"]["minecraft_version"] == "1.21.1"
    assert getattr(resolver._optimize, "_mmm_joint_reuse_optimizer", False) is True


def test_joint_reuse_optimizer_preserves_offline_host_path(monkeypatch) -> None:
    sentinel = SimpleNamespace(selected="offline")
    calls = []

    def original(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return sentinel

    resolver = SimpleNamespace(_optimize=original, SpecValidationError=RuntimeError)
    monkeypatch.setenv("MMM_ECOSYSTEM_DISCOVERY", "off")
    contract._install_joint_platform_optimizer(resolver)

    result = resolver._optimize(
        "offline mod",
        design=None,
        module_kinds=(),
        loader_constraint=None,
        version_constraint=None,
        target_research_fn=None,
    )
    assert result is sentinel
    assert len(calls) == 1


@dataclass(frozen=True)
class _Module:
    config: dict


def test_model_facing_reuse_config_drops_transport_manifests(monkeypatch, tmp_path) -> None:
    class DummyGenerator:
        def generate(self, project_root, *, module, **kwargs):
            return module.config

    module_owner = SimpleNamespace(CustomModuleGenerator=DummyGenerator)
    monkeypatch.setattr(
        contract,
        "_materialize_once",
        lambda *args, **kwargs: {"schema_version": "mmm/reuse-materialization-v1", "donors": [], "count": 0},
    )
    contract._install_reuse_materialization(module_owner)

    full_plan = {
        "target": {"minecraft_version": "1.21.1", "loader": "fabric"},
        "capabilities": [
            {"capability": "trade.transaction", "mode": "source_transplant", "source_id": "owner/repo"},
            {"capability": "trade.validation", "mode": "fresh", "source_id": ""},
        ],
    }
    output = DummyGenerator().generate(
        tmp_path,
        module=_Module(
            config={
                "name": "trade",
                "_approved_reuse_plan": {"large": "payload"},
                "_owned_reuse_plan": full_plan,
                "_reuse_materialization": {"large": "payload"},
                "_donor_source_excerpts": [{"large": "payload"}],
            }
        ),
    )

    assert output["name"] == "trade"
    assert output["_reuse_decisions"] == [
        {"capability": "trade.transaction", "mode": "source_transplant", "source_id": "owner/repo"},
        {"capability": "trade.validation", "mode": "fresh", "source_id": ""},
    ]
    assert output["_fresh_only_capabilities"] == ["trade.validation"]
    for key in contract._TRANSPORT_CONFIG_KEYS:
        assert key not in output


def test_donor_context_is_bounded_per_file_and_globally(tmp_path) -> None:
    files = []
    for index in range(4):
        path = tmp_path / f"F{index}.java"
        path.write_bytes(bytes([65 + index]) * 10_000)
        files.append({"path": str(path), "sha256": f"sha256:{index}"})
    receipt = {
        "donors": [
            {
                "repository": "owner/repo",
                "commit_sha": "a" * 40,
                "license_id": "MIT",
                "capability": "trade.transaction",
                "files": files,
            }
        ]
    }

    context = contract._materialized_donor_context(receipt, byte_budget=12_000, per_file_budget=4_000)

    assert len(context) == 3
    assert sum(len(item["content"].encode("utf-8")) for item in context) <= 12_000
    assert all(item["truncated"] for item in context)


def test_unverified_target_repository_is_not_a_reuse_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        transplant,
        "_repository_snapshot",
        lambda *args, **kwargs: {
            "license_id": "MIT",
            "commit_sha": "a" * 40,
            "source_url": "https://github.com/owner/repo",
            "blobs": {"src/main/java/TradeFeature.java": "b" * 40},
        },
    )
    monkeypatch.setattr(transplant, "_github_client", lambda token: _ClosableClient())
    monkeypatch.setattr(transplant, "_build_metadata_text", lambda *args, **kwargs: "")

    result = transplant.inspect_repository_slice(
        repository="owner/repo",
        capability="trade.transaction",
        adapter=SimpleNamespace(loader="fabric", minecraft_version="1.21.1"),
        discovery_client=SimpleNamespace(github_token=""),
    )
    assert result is None


def test_dependency_closure_is_not_truncated_to_first_512_java_paths(monkeypatch) -> None:
    blobs: dict[str, str] = {}
    target_path = "src/main/java/feature/TradeFeature.java"
    blobs[target_path] = "1" * 40
    for index in range(511):
        blobs[f"src/main/java/filler/Filler{index:03d}.java"] = f"{index:040x}"[-40:]
    tail_path = "src/main/java/z/TailDependency.java"
    blobs[tail_path] = "f" * 40
    monkeypatch.setattr(
        transplant,
        "_repository_snapshot",
        lambda *args, **kwargs: {
            "license_id": "MIT",
            "commit_sha": "a" * 40,
            "source_url": "https://github.com/owner/repo",
            "blobs": blobs,
        },
    )
    monkeypatch.setattr(transplant, "_github_client", lambda token: _ClosableClient())
    monkeypatch.setattr(
        transplant,
        "_build_metadata_text",
        lambda *args, **kwargs: "minecraft_version=1.21.1\nfabricloader",
    )

    def fake_blob(client, repository, blob_sha):
        if blob_sha == "1" * 40:
            return b"public class TradeFeature { TailDependency dep; public void trade() {} }"
        if blob_sha == "f" * 40:
            return b"public class TailDependency {}"
        return b"public class Filler {}"

    monkeypatch.setattr(transplant, "_fetch_blob_bytes", fake_blob)
    result = transplant.inspect_repository_slice(
        repository="owner/repo",
        capability="trade.transaction",
        adapter=SimpleNamespace(loader="fabric", minecraft_version="1.21.1"),
        discovery_client=SimpleNamespace(github_token=""),
    )

    assert result is not None
    assert tail_path in {item.path for item in result.files}


def test_parallel_blob_fetch_deduplicates_the_network_request(monkeypatch) -> None:
    raw = b"class TradeFeature {}"
    encoded = base64.b64encode(raw).decode("ascii")
    calls = 0
    calls_lock = threading.Lock()
    with transplant._BLOB_LOCK:
        transplant._BLOB_CACHE.clear()
        transplant._BLOB_INFLIGHT.clear()

    def fake_json(client, url, *, params=None):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {"encoding": "base64", "content": encoded}

    monkeypatch.setattr(transplant, "_github_json", fake_json)
    results: list[bytes] = []

    def worker() -> None:
        results.append(transplant._fetch_blob_bytes(object(), "owner/repo", "b" * 40))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == [raw] * 6
    assert calls == 1


def test_materialization_key_separates_capability_slices() -> None:
    donor = {"repository": "owner/repo", "commit_sha": "a" * 40}
    file_a = [{"path": "A.java", "blob_sha": "b" * 40, "sha256": "sha256:a"}]
    file_b = [{"path": "B.java", "blob_sha": "c" * 40, "sha256": "sha256:b"}]

    key_a = transplant._donor_materialization_key(
        {"capability": "trade.transaction"}, donor, file_a
    )
    key_b = transplant._donor_materialization_key(
        {"capability": "trade.validation"}, donor, file_b
    )
    assert key_a != key_b
