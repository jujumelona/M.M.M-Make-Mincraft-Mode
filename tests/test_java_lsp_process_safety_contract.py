from __future__ import annotations

import subprocess

import pytest

from minecraft_mod_ai import java_lsp
from minecraft_mod_ai.java_lsp_process_safety_contract import install


class _DeadProcess:
    returncode = 7
    stdin = object()

    def poll(self):
        return self.returncode


class _AliveProcess:
    returncode = None
    stdin = object()

    def poll(self):
        return None


def _bare_rpc(process):
    install(java_lsp)
    rpc = object.__new__(java_lsp._JsonRpcProcess)
    rpc.process = process
    rpc.messages = java_lsp.queue.Queue()
    rpc.stderr = java_lsp.deque(maxlen=30)
    rpc._next_id = 1
    rpc._mmm_request_lock = java_lsp.threading.RLock()
    rpc._mmm_write_lock = java_lsp.threading.RLock()
    rpc._mmm_reader_failure = None
    return rpc


def test_request_fails_immediately_when_reader_thread_has_crashed() -> None:
    rpc = _bare_rpc(_AliveProcess())
    rpc._mmm_reader_failure = ValueError("bad Content-Length")
    rpc.send = lambda _payload: None

    with pytest.raises(java_lsp.JDTLanguageServerError, match="stdout reader failed"):
        rpc.request("workspace/symbol", {"query": "x"}, timeout=10.0)


def test_send_rejects_already_dead_jdt_process() -> None:
    rpc = _bare_rpc(_DeadProcess())
    with pytest.raises(java_lsp.JDTLanguageServerError, match="exited before"):
        rpc.send({"jsonrpc": "2.0", "method": "exit", "params": {}})


def test_close_waits_after_kill(monkeypatch) -> None:
    install(java_lsp)

    class Process:
        def __init__(self):
            self.returncode = None
            self.wait_calls = 0
            self.killed = False
            self.terminated = False

        def poll(self):
            return None

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls <= 2:
                raise subprocess.TimeoutExpired("jdtls", timeout)
            self.returncode = -9
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    rpc = object.__new__(java_lsp._JsonRpcProcess)
    rpc.process = Process()
    rpc._reader = None
    rpc._error_reader = None
    rpc.request = lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError())
    rpc.notify = lambda *_args, **_kwargs: None

    rpc.close()
    assert rpc.process.terminated is True
    assert rpc.process.killed is True
    assert rpc.process.wait_calls == 3
