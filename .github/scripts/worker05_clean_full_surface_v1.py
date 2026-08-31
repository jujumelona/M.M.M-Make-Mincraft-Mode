from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# Output-budget serialization: known serialization failures mean the estimate is unavailable;
# programming errors must not be silently swallowed.
replace_once(
    "minecraft_mod_ai/generation_output_budget.py",
    '''        ).encode("utf-8")\n    except Exception:\n        return 0\n''',
    '''        ).encode("utf-8")\n    except (TypeError, ValueError, OverflowError, RecursionError):\n        return 0\n''',
)

# Prefill calibration is explicitly optional; retain the fail-open boundary but document it.
replace_once(
    "minecraft_mod_ai/llama_finish_reason_contract.py",
    '''        except Exception:\n            return ""\n''',
    '''        except Exception:  # noqa: BLE001 - optional prefill calibration must not block inference\n            return ""\n''',
)
replace_once(
    "minecraft_mod_ai/llama_finish_reason_contract.py",
    'raise RuntimeError("native llama-server returned an invalid completion choice")',
    'raise TypeError("native llama-server returned an invalid completion choice")',
)
replace_once(
    "minecraft_mod_ai/llama_finish_reason_contract.py",
    'raise RuntimeError("native llama-server returned no assistant message")',
    'raise TypeError("native llama-server returned no assistant message")',
)

# Remove the legacy wall-clock heartbeat owner from the adapter itself. The semantic SSE
# watchdog is the sole liveness owner; no request should create a blind reporter thread.
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '''    heartbeat_seconds = _positive_env_float(\n        "MMM_LLAMA_COMPLETION_HEARTBEAT_SECONDS",\n        _DEFAULT_COMPLETION_HEARTBEAT_SECONDS,\n    )\n    started = time.monotonic()\n    stop = threading.Event()\n''',
    "",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '''\n    def report_pending() -> None:\n        while not stop.wait(heartbeat_seconds):\n            print(\n                "llama server: completion pending",\n                f" elapsed={time.monotonic() - started:.1f}s",\n                f" input_chars={input_chars}",\n                f" max_tokens={max_tokens}",\n                f" tools={tool_count}",\n                sep="",\n                flush=True,\n            )\n\n    reporter = threading.Thread(\n        target=report_pending,\n        name="mmm-llama-completion-liveness",\n        daemon=True,\n    )\n    reporter.start()\n''',
    "\n",
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '''    except httpx.TimeoutException as exc:\n        raise RuntimeError(\n            "native llama-server completion made no readable progress for "\n            f"{read_timeout:.0f}s"\n        ) from exc\n    finally:\n        stop.set()\n        reporter.join(timeout=0.2)\n''',
    '''    except httpx.TimeoutException as exc:\n        raise RuntimeError(\n            "native llama-server completion made no readable progress for "\n            f"{read_timeout:.0f}s"\n        ) from exc\n''',
)
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '_DEFAULT_COMPLETION_HEARTBEAT_SECONDS = 15.0\n',
    "",
)
for message in (
    "assistant-prefill calibration returned invalid JSON",
    "assistant-prefill calibration returned an invalid choice",
    "assistant-prefill calibration returned no message",
    "tool schema lacks function metadata",
    "named tool_choice lacks function metadata",
    "native llama-server returned an invalid completion choice",
    "native llama-server returned no assistant message",
):
    replace_once(
        "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
        f'raise RuntimeError("{message}")',
        f'raise TypeError("{message}")',
    )
replace_once(
    "minecraft_mod_ai/model_adapters/llama_cpp_adapter.py",
    '''    try:\n        body = str(response.text)\n    except Exception:\n        return ""\n''',
    '''    try:\n        body = str(response.text)\n    except (AttributeError, httpx.HTTPError, RuntimeError, TypeError, ValueError):\n        return ""\n''',
)

# Runtime tuning: narrow predictable probe failures; explicitly mark the few boundaries that
# deliberately isolate optional benchmark candidates from each other.
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''        except Exception:\n            return\n\n    reader = threading.Thread(\n''',
    '''        except (OSError, ValueError):\n            return\n\n    reader = threading.Thread(\n''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''    try:\n        return json.loads(raw)\n    except Exception:\n        return {}\n''',
    '''    try:\n        return json.loads(raw)\n    except json.JSONDecodeError:\n        return {}\n''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''        gpu_probe_succeeded = True\n    except Exception:\n        pass\n\n    ram_available = 0\n''',
    '''        gpu_probe_succeeded = True\n    except (OSError, subprocess.SubprocessError, IndexError, ValueError):\n        gpu_probe_succeeded = False\n\n    ram_available = 0\n''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''                ram_probe_succeeded = True\n                break\n    except Exception:\n        pass\n\n    gpu_free_override =''',
    '''                ram_probe_succeeded = True\n                break\n    except (OSError, UnicodeError, IndexError, ValueError):\n        ram_probe_succeeded = False\n\n    gpu_free_override =''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''    except Exception as exc:\n        return autotune_module.ProbeResult(\n            variant=variant,\n''',
    '''    except Exception as exc:  # noqa: BLE001 - benchmark failures become comparable ProbeResults\n        return autotune_module.ProbeResult(\n            variant=variant,\n''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''                except Exception as exc:\n                    failures.append(f"p{slots}: {type(exc).__name__}: {exc}")\n''',
    '''                except Exception as exc:  # noqa: BLE001 - isolate sequential launch candidates\n                    failures.append(f"p{slots}: {type(exc).__name__}: {exc}")\n''',
)
replace_once(
    "minecraft_mod_ai/llama_server_runtime_tuning.py",
    '''                    except Exception:\n                        receipt = None\n''',
    '''                    except (json.JSONDecodeError, TypeError):\n                        receipt = None\n''',
)
replace_once(
    "minecraft_mod_ai/llama_tuning_pipeline.py",
    '''            except Exception:\n                hardware = ""\n''',
    '''            except Exception:  # noqa: BLE001 - optional hardware identity must not block portable tuning\n                hardware = ""\n''',
)

# Tests should state the exact failure contract and use protocol-correct context-manager types.
replace_once(
    "tests/test_llama_server_streaming.py",
    '''import httpx\n\nfrom minecraft_mod_ai''',
    '''import httpx\nimport pytest\n\nfrom minecraft_mod_ai''',
)
replace_once(
    "tests/test_llama_server_streaming.py",
    '''    try:\n        _strict_server_generate(adapter, request, "http://127.0.0.1:8910/v1")\n    except Exception as exc:\n        assert "stream ended before the [DONE] marker" in str(exc)\n    else:\n        raise AssertionError("truncated SSE stream must fail closed")\n''',
    '''    with pytest.raises(RuntimeError, match=r"stream ended before the \\[DONE\\] marker"):\n        _strict_server_generate(adapter, request, "http://127.0.0.1:8910/v1")\n''',
)
replace_once(
    "tests/test_llama_stream_tool_completion.py",
    '''import json\nfrom typing import Any\n''',
    '''import json\nfrom types import TracebackType\nfrom typing import Any, Self\n''',
)
replace_once(
    "tests/test_llama_stream_tool_completion.py",
    '''    def __enter__(self) -> _FakeStreamResponse:\n        return self\n\n    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:\n        return None\n''',
    '''    def __enter__(self) -> Self:\n        return self\n\n    def __exit__(\n        self,\n        exc_type: type[BaseException] | None,\n        exc: BaseException | None,\n        traceback: TracebackType | None,\n    ) -> None:\n        return None\n''',
)
replace_once(
    "tests/test_llama_tool_stream_transport.py",
    '''class _StreamResponse:\n    status_code = 200\n    headers: dict[str, str] = {}\n\n    def __init__(self, lines: list[str]) -> None:\n        self._lines = lines\n''',
    '''class _StreamResponse:\n    status_code = 200\n\n    def __init__(self, lines: list[str]) -> None:\n        self.headers: dict[str, str] = {}\n        self._lines = lines\n''',
)

# The final gate is the only temporary Worker05 workflow allowed while this cleanup runs.
# This cleanup workflow and script delete themselves in the production commit.
(ROOT / ".github/workflows/worker05-clean-full-surface-v1.yml").unlink(missing_ok=True)
Path(__file__).unlink(missing_ok=True)
