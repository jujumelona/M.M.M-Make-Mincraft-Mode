from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "minecraft_mod_ai/llama_server_hardware_policy.py"
TEST = ROOT / "tests/test_llama_server_hardware_policy.py"


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path.relative_to(ROOT)}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Keep server-launch policy testable outside the runtime monkeypatch installer.
insert_before = "\ndef install(autotune_module: Any) -> None:\n"
helper = '''\ndef _apply_hardware_launch_policy(args: list[str]) -> list[str]:\n    """Apply managed llama-server launch policy without enabling unused endpoints."""\n\n    try:\n        index = args.index("--gpu-layers")\n        args[index + 1] = "auto"\n    except (ValueError, IndexError):\n        pass\n    if "--parallel" not in args and "-np" not in args:\n        args.extend(["--parallel", "1"])\n    if _auxiliary_native_telemetry_enabled():\n        if "--metrics" not in args:\n            args.append("--metrics")\n        if "--slots" not in args:\n            args.append("--slots")\n    return args\n\n\ndef install(autotune_module: Any) -> None:\n'''
replace_once(POLICY, insert_before, helper)
replace_once(
    POLICY,
    '''            args = original_base(binary, model_path, config, port)\n            try:\n                index = args.index("--gpu-layers")\n                args[index + 1] = "auto"\n            except (ValueError, IndexError):\n                pass\n            if "--parallel" not in args and "-np" not in args:\n                args.extend(["--parallel", "1"])\n            if "--metrics" not in args:\n                args.append("--metrics")\n            if "--slots" not in args:\n                args.append("--slots")\n            return args\n''',
    '''            args = original_base(binary, model_path, config, port)\n            return _apply_hardware_launch_policy(args)\n''',
)
replace_once(
    POLICY,
    "        adaptive_base_args._mmm_native_telemetry_endpoints = True  # type: ignore[attr-defined]\n",
    "        adaptive_base_args._mmm_auxiliary_telemetry_opt_in = True  # type: ignore[attr-defined]\n",
)

with TEST.open("a", encoding="utf-8") as handle:
    handle.write(
        '''\n\ndef test_hardware_launch_policy_does_not_enable_auxiliary_endpoints_by_default(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    args = ["llama-server", "--gpu-layers", "all"]\n\n    result = policy._apply_hardware_launch_policy(args)\n\n    assert result is args\n    assert result[result.index("--gpu-layers") + 1] == "auto"\n    assert result[result.index("--parallel") + 1] == "1"\n    assert "--metrics" not in result\n    assert "--slots" not in result\n\n\ndef test_hardware_launch_policy_enables_auxiliary_endpoints_only_on_opt_in(monkeypatch) -> None:\n    monkeypatch.setenv("MMM_LLAMA_AUXILIARY_TELEMETRY", "true")\n    args = ["llama-server", "--gpu-layers", "all"]\n\n    result = policy._apply_hardware_launch_policy(args)\n\n    assert result.count("--metrics") == 1\n    assert result.count("--slots") == 1\n\n\ndef test_hardware_launch_policy_preserves_explicit_operator_telemetry_flags(monkeypatch) -> None:\n    monkeypatch.delenv("MMM_LLAMA_AUXILIARY_TELEMETRY", raising=False)\n    args = ["llama-server", "--metrics", "--slots"]\n\n    result = policy._apply_hardware_launch_policy(args)\n\n    assert result.count("--metrics") == 1\n    assert result.count("--slots") == 1\n'''
    )

(ROOT / ".github/workflows/worker05-aux-telemetry-flags-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
