from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one anchor, got {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Temporary skills must use the canonical trajectory-memory owner, not a copied late-bound alias.
replace_once(
    "minecraft_mod_ai/temporary_skill_contract.py",
    """from .remote_trajectory_store import (\n    flush_remote_outbox,\n    hydrate_remote_cache,\n    queue_remote_record,\n    remote_configured,\n)\nfrom .trajectory_memory import (\n""",
    """from .remote_trajectory_store import (\n    flush_remote_outbox,\n    hydrate_remote_cache,\n    queue_remote_record,\n    remote_configured,\n)\nfrom . import trajectory_memory as _trajectory_memory\nfrom .trajectory_memory import (\n""",
)
replace_once(
    "minecraft_mod_ai/temporary_skill_contract.py",
    "    relevant_trajectories,\n",
    "",
)
replace_once(
    "minecraft_mod_ai/temporary_skill_contract.py",
    "    records = relevant_trajectories(\n",
    "    records = _trajectory_memory.relevant_trajectories(\n",
)

# SQLite candidate acceleration must preserve the v3 execution-context hard gate.
replace_once(
    "minecraft_mod_ai/research_memory_performance.py",
    """        router: Any | None = None,\n        limit: int = 6,\n    ) -> list[dict[str, Any]]:\n""",
    """        router: Any | None = None,\n        limit: int = 6,\n        current_context: Mapping[str, Any] | None = None,\n    ) -> list[dict[str, Any]]:\n""",
)
replace_once(
    "minecraft_mod_ai/research_memory_performance.py",
    """                return original(base, query, task_class=task_class, router=router, limit=limit)\n""",
    """                return original(\n                    base,\n                    query,\n                    task_class=task_class,\n                    router=router,\n                    limit=limit,\n                    current_context=current_context,\n                )\n""",
)
replace_once(
    "minecraft_mod_ai/research_memory_performance.py",
    """        for row in rows:\n            if str(row.get(\"task_class\", \"\")) not in {task_class, \"general\"}:\n                continue\n            rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)\n""",
    """        for row in rows:\n            if str(row.get(\"task_class\", \"\")) not in {task_class, \"general\"}:\n                continue\n            if not tm._execution_context_compatible(row, current_context):\n                continue\n            rendered = json.dumps(row, ensure_ascii=False, sort_keys=True)\n""",
)

# Hotpath context binding must forward the same state rather than silently dropping it.
replace_once(
    "minecraft_mod_ai/runtime_hotpath_consolidation.py",
    """        router: Any | None = None,\n        limit: int = 6,\n    ) -> list[dict[str, Any]]:\n        token = _MEMORY_BASE.set(str(Path(base).expanduser().resolve()))\n""",
    """        router: Any | None = None,\n        limit: int = 6,\n        current_context: Mapping[str, Any] | None = None,\n    ) -> list[dict[str, Any]]:\n        token = _MEMORY_BASE.set(str(Path(base).expanduser().resolve()))\n""",
)
replace_once(
    "minecraft_mod_ai/runtime_hotpath_consolidation.py",
    """                router=router,\n                limit=limit,\n            )\n""",
    """                router=router,\n                limit=limit,\n                current_context=current_context,\n            )\n""",
)

# The late bottleneck installer only needs to rebind append-by-value now. Retrieval stays module-owned.
replace_once(
    "minecraft_mod_ai/research_bottleneck_runtime.py",
    """    # temporary_skill_contract imported these functions by value before this late\n    # bootstrap entry. Rebind only its module globals; do not add another work owner.\n    try:\n        from . import temporary_skill_contract as temporary\n        temporary.append_trajectory = indexed_append\n        temporary.relevant_trajectories = indexed_relevant\n    except Exception:\n        pass\n""",
    """    # temporary_skill_contract still imports append_trajectory by value. Retrieval\n    # is deliberately module-owned so execution-context filtering cannot be bypassed\n    # by a stale late-bootstrap alias.\n    try:\n        from . import temporary_skill_contract as temporary\n        temporary.append_trajectory = indexed_append\n    except Exception:\n        pass\n""",
)

# Tests patch the canonical owner just like production now does.
replace_once(
    "tests/test_temporary_skill_reuse.py",
    '    monkeypatch.setattr(contract, "relevant_trajectories", fake_relevant)\n',
    '    monkeypatch.setattr(contract._trajectory_memory, "relevant_trajectories", fake_relevant)\n',
)

p = Path("tests/test_runtime_json_gap_regression.py")
text = p.read_text(encoding="utf-8")
insert = '''\n\ndef test_runtime_trajectory_retrieval_keeps_execution_context_contract():\n    import inspect\n\n    from minecraft_mod_ai import temporary_skill_contract, trajectory_memory\n\n    assert "current_context" in inspect.signature(trajectory_memory.relevant_trajectories, follow_wrapped=False).parameters\n    assert temporary_skill_contract._trajectory_memory is trajectory_memory\n    assert "relevant_trajectories" not in temporary_skill_contract.__dict__\n'''
if "test_runtime_trajectory_retrieval_keeps_execution_context_contract" not in text:
    text += insert
p.write_text(text, encoding="utf-8")
