from __future__ import annotations

import threading

from minecraft_mod_ai.model_concurrency import ReentrantReadWriteLock


def test_concurrent_read_to_write_upgrades_do_not_deadlock() -> None:
    lock = ReentrantReadWriteLock()
    barrier = threading.Barrier(2)
    completed: list[int] = []
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            lock.acquire_read()
            try:
                barrier.wait(timeout=2.0)
                lock.acquire()
                try:
                    completed.append(index)
                finally:
                    lock.release()
            finally:
                lock.release_read()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert sorted(completed) == [0, 1]
