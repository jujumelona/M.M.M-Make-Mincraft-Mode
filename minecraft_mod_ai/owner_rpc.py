"""One persistent JSON-lines process boundary with deterministic retirement.

EOF and responses wake the caller; no PID polling, readiness probes or retries.
A timeout retires the process so a late response cannot authorize another request.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Callers own their public/idle timeout. Some verifier paths intentionally pass a
# longer hard budget so a cold JDT owner can finish startup. Honor that explicit
# budget up to a transport safety ceiling instead of silently truncating it to 90s.
_MAX_OWNER_REQUEST_SECONDS = 600.0


class OwnerRPCError(RuntimeError):
    pass


class OwnerRPC:
    def __init__(self, command: Sequence[str], *, cwd: Path | None = None) -> None:
        if not command or isinstance(command, str):
            raise ValueError('Owner command must be a nonempty argument sequence')
        self.process = subprocess.Popen(
            list(command), cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='strict',
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        self._responses: queue.Queue[Any] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=40)
        self._stdout_noise: deque[str] = deque(maxlen=40)
        self._lock = threading.RLock()
        self._closed = False
        self._sequence = 0
        self._reader = threading.Thread(target=self._read, daemon=True, name='mmm-owner-rpc')
        self._errors = threading.Thread(target=self._read_errors, daemon=True, name='mmm-owner-stderr')
        self._reader.start()
        self._errors.start()

    def _decode_response_line(self, line: str) -> dict[str, Any] | OwnerRPCError | None:
        candidate = line.strip()
        if not candidate:
            return None
        # Eclipse/Equinox and the JVM may write startup diagnostics to the inherited
        # stdout before OwnerApplication redirects System.out. Protocol replies are JSON objects.
        if not candidate.startswith('{'):
            self._stdout_noise.append(candidate[-2000:])
            return None
        try:
            message = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return OwnerRPCError(
                f'Owner protocol error: {exc}; stdout frame prefix={candidate[:200]!r}'
            )
        if not isinstance(message, dict):
            return OwnerRPCError('Owner protocol error: response must be an object')
        return message

    def _queue_response_line(self, line: str) -> bool:
        message = self._decode_response_line(line)
        if message is None:
            return False
        self._responses.put(message)
        return isinstance(message, OwnerRPCError)

    def _read(self) -> None:
        try:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                if self._queue_response_line(line):
                    return
        except (TypeError, ValueError, OSError, UnicodeError) as exc:
            self._responses.put(OwnerRPCError(f'Owner protocol error: {exc}'))
        finally:
            self._responses.put(OwnerRPCError('Owner process closed its response stream'))

    def _read_errors(self) -> None:
        try:
            assert self.process.stderr is not None
            for line in self.process.stderr:
                self._stderr.append(line.rstrip()[-2000:])
        except (OSError, UnicodeError):
            return

    def _timeout_error(self, method: str, timeout: float) -> OwnerRPCError:
        return self._mmm__timeout_error_impl(method, timeout)

    def _mmm__timeout_error_impl(self, method: str, timeout: float) -> OwnerRPCError:
        noise = list(self._stdout_noise)
        suffix = f'; non-protocol stdout tail={noise!r}' if noise else ''
        stderr = list(self._stderr)
        if stderr:
            suffix += f'; backend stderr tail={stderr!r}'
        return OwnerRPCError(f'Owner {method} timed out after {timeout}s{suffix}')

    def _raise_transport_failure(
        self,
        method: str,
        timeout: float,
        exc: BaseException,
    ) -> None:
        self.close()
        if isinstance(exc, queue.Empty):
            raise self._timeout_error(method, timeout) from exc
        if isinstance(exc, OwnerRPCError):
            raise exc
        raise OwnerRPCError(f'Owner transport failed: {exc}') from exc

    def request(self, method: str, params: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
        if timeout <= 0:
            raise ValueError('Owner request timeout must be positive')
        timeout = min(float(timeout), _MAX_OWNER_REQUEST_SECONDS)
        with self._lock:
            if self._closed:
                raise OwnerRPCError('Owner process is closed')
            self._sequence += 1
            request_id = self._sequence
            wire = json.dumps({'id': request_id, 'method': method, 'params': dict(params)},
                              allow_nan=False, separators=(',', ':'))
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(wire + '\n')
                self.process.stdin.flush()
                response = self._responses.get(timeout=timeout)
                if isinstance(response, Exception):
                    raise response
                if response.get('id') != request_id:
                    raise OwnerRPCError('Owner protocol response ID mismatch')
                if 'error' in response:
                    raise OwnerRPCError(f"Owner {method} failed: {response['error']}")
                result = response.get('result')
                if not isinstance(result, dict):
                    raise OwnerRPCError('Owner protocol result must be an object')
                return result
            except (queue.Empty, OSError, ValueError, OwnerRPCError) as exc:
                self._raise_transport_failure(method, timeout, exc)
                raise AssertionError('unreachable')

    def _process_exited(self, timeout: float) -> bool:
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def _kill_and_wait(self) -> None:
        self.process.kill()
        self._process_exited(2)

    def _terminate_or_kill(self) -> None:
        self.process.terminate()
        if not self._process_exited(2):
            self._kill_and_wait()

    def _retire_process(self) -> None:
        if not self._process_exited(2):
            self._terminate_or_kill()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                self._retire_process()
            finally:
                self._reader.join(timeout=2)
                self._errors.join(timeout=2)
                for stream in (self.process.stdout, self.process.stderr):
                    if stream is not None:
                        stream.close()

    def __enter__(self) -> OwnerRPC:  # noqa: PYI034 -- Python 3.10 support
        if self._closed:
            raise OwnerRPCError('Owner process is closed')
        return self

    def __exit__(self, *_args: object) -> bool:
        self.close()
        return False
