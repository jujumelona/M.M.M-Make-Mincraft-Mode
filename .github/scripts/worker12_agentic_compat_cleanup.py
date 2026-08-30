from pathlib import Path

path = Path("minecraft_mod_ai/agentic_optimization_contract.py")
text = path.read_text(encoding="utf-8")
old = '''    evidence remain authoritative and are intentionally not replaced here.\n    """\n    _install_repair_search_and_memory(repair_module)\n'''
new = '''    evidence remain authoritative and are intentionally not replaced here.\n    """\n    # Kept as a keyword-only compatibility hook for the shared installer contract.\n    # This module intentionally does not mutate planner authority.\n    del complete_planner_module\n    _install_repair_search_and_memory(repair_module)\n'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one installer compatibility anchor, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("worker12 agentic installer compatibility cleanup applied")
