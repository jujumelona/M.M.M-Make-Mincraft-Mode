from __future__ import annotations

from functools import wraps
from typing import Any

_SYNC_PLAN_TEMP_TABLES = (
    "affected_nodes",
    "changed_nodes",
    "desired_edges",
    "desired_tasks",
)


def _drop_sync_plan_temp_tables(ledger: Any) -> None:
    """Keep sync_plan TEMP state bounded on reusable SQLite connections."""

    with ledger._connect() as connection:
        for table_name in _SYNC_PLAN_TEMP_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.commit()


def install(work_graph_module: Any) -> None:
    """Preserve only reusable-connection cleanup required by the event scheduler.

    Generation scheduling is now event driven and no longer performs high-frequency
    full-node polling. Wrapping ``task`` and ``claim_ready`` with a synthetic snapshot
    therefore adds invalidation work to the hot path without removing meaningful I/O.
    The only remaining responsibility here is cleaning sync_plan TEMP tables because
    scheduler_parallel_safety_contract deliberately reuses one SQLite connection per
    ledger/thread.
    """

    ledger_cls = work_graph_module.DurableWorkLedger
    current_sync_plan = ledger_cls.sync_plan
    if getattr(current_sync_plan, "_mmm_reusable_connection_sync_plan", False):
        return

    @wraps(current_sync_plan)
    def sync_plan(self: Any, plan: Any):
        _drop_sync_plan_temp_tables(self)
        try:
            return current_sync_plan(self, plan)
        finally:
            # sqlite3.Connection.__exit__ has committed/rolled back the original
            # transaction before cleanup, so this cannot partially commit sync_plan.
            _drop_sync_plan_temp_tables(self)

    sync_plan._mmm_reusable_connection_sync_plan = True  # type: ignore[attr-defined]
    sync_plan.__wrapped__ = current_sync_plan  # type: ignore[attr-defined]
    ledger_cls.sync_plan = sync_plan


__all__ = ["install"]
