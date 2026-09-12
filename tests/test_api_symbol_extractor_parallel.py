from __future__ import annotations

import threading
import time
from types import MethodType

from minecraft_mod_ai.api_symbol_extractor import MinecraftAPIExtractor
from minecraft_mod_ai.implementation_identity import compute_content_hash


def test_extract_from_maven_downloads_independent_artifacts_concurrently(monkeypatch):
    payloads = {
        "https://example.test/a.jar": b"a",
        "https://example.test/b.jar": b"b",
        "https://example.test/c.jar": b"c",
    }
    lock = threading.Lock()
    active = 0
    maximum = 0

    class Response:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self.data

    def fake_urlopen(url, timeout):
        nonlocal active, maximum
        assert timeout == 60
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        data = payloads[url]
        with lock:
            active -= 1
        return Response(data)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    extractor = MinecraftAPIExtractor()

    def fake_extract(self, path, mappings_file=None):
        del self, mappings_file
        value = path.read_bytes().decode("ascii")
        return {value: value}

    extractor.extract_from_minecraft_jar = MethodType(fake_extract, extractor)
    artifacts = [
        {"url": url, "sha256": compute_content_hash(data)}
        for url, data in payloads.items()
    ]

    assert extractor.extract_from_maven("ignored", artifacts=artifacts) == {
        "a": "a",
        "b": "b",
        "c": "c",
    }
    assert maximum > 1
