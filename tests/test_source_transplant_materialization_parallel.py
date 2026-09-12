from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path

import pytest

from minecraft_mod_ai import source_transplant as st


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _file(path: str, blob_sha: str, payload: bytes) -> st.DonorFile:
    return st.DonorFile(
        path=path,
        blob_sha=blob_sha,
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        symbols=(),
    )


def _donor(files: tuple[st.DonorFile, ...]) -> st.DonorSlice:
    return st.DonorSlice(
        capability="parallel-materialization-test",
        repository="owner/repo",
        commit_sha="1" * 40,
        license_id="MIT",
        source_url="https://github.com/owner/repo",
        target_compatibility="exact",
        files=files,
        seed_files=(),
        source_symbols=(),
        required_dependencies=(),
        donor_tests=(),
        confidence=1.0,
    )


def _plan() -> dict[str, object]:
    return {
        "capabilities": [
            {
                "mode": "source_transplant",
                "capability": "parallel-materialization-test",
                "donor": {},
            }
        ]
    }


def test_materialization_parallelizes_fetch_but_preserves_manifest_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payloads = {
        "a" * 40: b"first",
        "b" * 40: b"second",
        "c" * 40: b"third",
    }
    donor = _donor(
        (
            _file("src/First.java", "a" * 40, payloads["a" * 40]),
            _file("src/Second.java", "b" * 40, payloads["b" * 40]),
            _file("src/Third.java", "c" * 40, payloads["c" * 40]),
        )
    )
    fake_client = _FakeClient()
    active = 0
    peak = 0
    lock = threading.Lock()
    delays = {"a" * 40: 0.08, "b" * 40: 0.04, "c" * 40: 0.01}

    def fake_fetch(_client: object, _repository: str, blob_sha: str) -> bytes:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(delays[blob_sha])
            return payloads[blob_sha]
        finally:
            with lock:
                active -= 1

    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_MATERIALIZE_DOWNLOAD_WORKERS", "3")
    monkeypatch.setattr(st, "validated_reuse_donor", lambda _decision: donor)
    monkeypatch.setattr(st, "_github_client", lambda _token: fake_client)
    monkeypatch.setattr(st, "_fetch_blob_bytes", fake_fetch)

    result = st.materialize_source_slices(tmp_path, _plan())

    assert peak >= 2
    assert result["count"] == 1
    manifest = result["donors"][0]
    assert [entry["source_path"] for entry in manifest["files"]] == [
        "src/First.java",
        "src/Second.java",
        "src/Third.java",
    ]
    assert [Path(entry["path"]).read_bytes() for entry in manifest["files"]] == [
        b"first",
        b"second",
        b"third",
    ]
    assert fake_client.closed is True


def test_materialization_hash_failure_writes_no_partial_donor_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    good = b"good"
    expected = b"expected"
    donor = _donor(
        (
            _file("src/Good.java", "d" * 40, good),
            _file("src/Bad.java", "e" * 40, expected),
        )
    )
    fake_client = _FakeClient()
    payloads = {"d" * 40: good, "e" * 40: b"wrong"}

    monkeypatch.setenv("MMM_SOURCE_TRANSPLANT_MATERIALIZE_DOWNLOAD_WORKERS", "2")
    monkeypatch.setattr(st, "validated_reuse_donor", lambda _decision: donor)
    monkeypatch.setattr(st, "_github_client", lambda _token: fake_client)
    monkeypatch.setattr(
        st,
        "_fetch_blob_bytes",
        lambda _client, _repository, blob_sha: payloads[blob_sha],
    )

    with pytest.raises(st.SourceTransplantError, match="hash mismatch"):
        st.materialize_source_slices(tmp_path, _plan())

    assert not [path for path in tmp_path.rglob("*") if path.is_file()]
    assert fake_client.closed is True
