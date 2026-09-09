from pathlib import Path

import httpx

from minecraft_mod_ai import java_diagnostics_fallback_contract as fallback
from minecraft_mod_ai import llama_exact_context as exact


def test_exact_context_probe_retries_transient_connect_error(monkeypatch):
    calls = {"count": 0}
    class Response:
        def raise_for_status(self):
            return None
    def request():
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ConnectError("not ready", request=httpx.Request("GET", "http://localhost/props"))
        return Response()
    monkeypatch.setattr(exact.time, "sleep", lambda _seconds: None)
    assert exact._request_with_transient_retry(request, attempts=3) is not None
    assert calls["count"] == 3


def test_exact_context_probe_does_not_retry_non_transient_http(monkeypatch):
    calls = {"count": 0}
    def request():
        calls["count"] += 1
        response = httpx.Response(400, request=httpx.Request("GET", "http://localhost/props"))
        raise httpx.HTTPStatusError("bad request", request=response.request, response=response)
    monkeypatch.setattr(exact.time, "sleep", lambda _seconds: None)
    try:
        exact._request_with_transient_retry(request, attempts=3)
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("non-transient HTTP errors must fail fast")
    assert calls["count"] == 1


def test_jdt_unavailable_falls_back_to_gradle_failure(monkeypatch, tmp_path):
    class Service:
        workspace_root = tmp_path
        def _existing_dir(self, value):
            return Path(value)
        def java_diagnostics(self, project_root, relative_files=None, timeout_seconds=60):
            return {"status": "UNAVAILABLE", "error": "jdt down", "diagnostics": {}}
    class Report:
        passed = False
        error = "Gradle build failed."
        def to_dict(self):
            return {"status": "FAIL", "error": self.error}
    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass
        def build(self, _root, *, run_gametest):
            assert run_gametest is False
            return Report()
    monkeypatch.setattr(fallback, "GradleRunner", Runner)
    fallback.install(Service)
    receipt = Service().java_diagnostics(str(tmp_path))
    assert receipt["verifier"] == "gradle_build_fallback"
    assert receipt["diagnostics"][0]["code"] == "GRADLE_BUILD_FAILED"
