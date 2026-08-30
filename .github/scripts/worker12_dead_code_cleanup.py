from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one cleanup anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "minecraft_mod_ai/agentic_optimization_contract.py",
    "def install(*, complete_planner_module: Any, repair_module: Any, work_graph_module: Any) -> None:\n",
    "def install(*, repair_module: Any, work_graph_module: Any) -> None:\n",
)

replace_once(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "        complete_orchestrator,\n        complete_planner,\n        custom_module_generator,\n",
    "        complete_orchestrator,\n        custom_module_generator,\n",
)

replace_once(
    "minecraft_mod_ai/runtime_bootstrap.py",
    "    agentic_optimization_contract.install(\n        complete_planner_module=complete_planner,\n        repair_module=repair_engine,\n        work_graph_module=work_graph,\n    )\n",
    "    agentic_optimization_contract.install(\n        repair_module=repair_engine,\n        work_graph_module=work_graph,\n    )\n",
)

print("worker12 dead integration parameter removed")
