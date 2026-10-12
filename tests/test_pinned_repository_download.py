from __future__ import annotations

import hashlib
import io
import zipfile

import httpx

from minecraft_mod_ai import source_transplant as transplant


def test_commit_archive_populates_verified_blob_cache_without_api_fanout():
    source = b"package demo; public class PlanetTravel {}"
    blob_sha = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("project-commit/src/main/java/demo/PlanetTravel.java", source)
    requests = []

    def download(request):
        requests.append(request)
        assert request.url.host == "codeload.github.com"
        assert not request.headers.get("Authorization")
        return httpx.Response(200, content=stream.getvalue())

    with httpx.Client(transport=httpx.MockTransport(download), headers={"Authorization": "Bearer secret-test"}) as client:
        assert transplant._prefetch_pinned_archive(
            client, "example/space", "a" * 40,
            {"src/main/java/demo/PlanetTravel.java": blob_sha},
        ) == 1
        assert transplant._fetch_blob_bytes(client, "example/space", blob_sha) == source
    assert len(requests) == 1


def test_archive_hash_mismatch_cannot_enter_source_cache():
    import pytest

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("project/src/main/java/demo/Fake.java", b"corrupted source")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=stream.getvalue()))) as client:
        with pytest.raises(transplant.SourceTransplantError, match="hash mismatch"):
            transplant._prefetch_pinned_archive(client, "example/corrupt", "b" * 40,
                                                {"src/main/java/demo/Fake.java": "c" * 40})
