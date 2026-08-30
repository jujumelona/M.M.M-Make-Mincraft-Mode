from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARDWARE = ROOT / "minecraft_mod_ai/llama_server_hardware_policy.py"
TEST = ROOT / "tests/test_llama_server_hardware_policy.py"
text = HARDWARE.read_text(encoding="utf-8")


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"hardware policy cleanup anchor count={count}: {old[:90]!r}")
    text = text.replace(old, new, 1)


replace_once(
    '        raise ValueError("named tool_choice requires function metadata")',
    '        raise TypeError("named tool_choice requires function metadata")',
)
replace_once(
    '''        elif value is not None:\n            try:\n                total += len(json.dumps(value, ensure_ascii=False))\n            except Exception:\n                pass\n''',
    '''        elif value is not None:\n            try:\n                total += len(json.dumps(value, ensure_ascii=False))\n            except (TypeError, ValueError, RecursionError):\n                # Telemetry sizing must never make inference fail on an exotic payload.\n                total += len(str(value))\n''',
)
replace_once(
    '''    except Exception:\n        return None\n\n\ndef _slot_snapshot''',
    '''    except Exception as exc:  # noqa: BLE001 - optional metrics endpoint boundary\n        print(\n            "llama server: metrics snapshot unavailable",\n            f" error={type(exc).__name__}",\n            flush=True,\n        )\n        return None\n\n\ndef _slot_snapshot''',
)
replace_once(
    '''    except Exception:\n        return None\n\n\ndef _telemetry_totals''',
    '''    except Exception as exc:  # noqa: BLE001 - optional slot endpoint boundary\n        print(\n            "llama server: slot snapshot unavailable",\n            f" error={type(exc).__name__}",\n            flush=True,\n        )\n        return None\n\n\ndef _telemetry_totals''',
)
replace_once(
    '''            try:\n                metrics_after = _metrics_snapshot(client, server_url)\n                _commit_metrics_delta(metrics_before, metrics_after)\n            except Exception:\n                pass\n''',
    '''            try:\n                metrics_after = _metrics_snapshot(client, server_url)\n                _commit_metrics_delta(metrics_before, metrics_after)\n            except Exception as telemetry_exc:  # noqa: BLE001 - best-effort failure telemetry\n                print(\n                    "llama server: failure telemetry unavailable",\n                    f" error={type(telemetry_exc).__name__}",\n                    flush=True,\n                )\n''',
)
replace_once(
    '''            try:\n                client.close()\n            except Exception:\n                pass\n''',
    '''            try:\n                client.close()\n            except Exception as close_exc:  # noqa: BLE001 - transport cleanup boundary\n                print(\n                    "llama server: client close failed",\n                    f" error={type(close_exc).__name__}",\n                    flush=True,\n                )\n''',
)
replace_once(
    '''                except Exception:\n                    local_exclusive_image = False\n''',
    '''                except Exception as exc:  # noqa: BLE001 - optional registry lookup boundary\n                    print(\n                        "llama server: image role lookup unavailable",\n                        f" error={type(exc).__name__}",\n                        flush=True,\n                    )\n                    local_exclusive_image = False\n''',
)
HARDWARE.write_text(text, encoding="utf-8")

test_text = TEST.read_text(encoding="utf-8")
test_text = test_text.replace(
    'assert hardware._auxiliary_native_telemetry_enabled() is False',
    'assert policy._auxiliary_native_telemetry_enabled() is False',
)
test_text = test_text.replace(
    'assert hardware._auxiliary_native_telemetry_enabled() is True',
    'assert policy._auxiliary_native_telemetry_enabled() is True',
)
if "hardware._auxiliary_native_telemetry_enabled" in test_text:
    raise SystemExit("stale hardware alias remains in telemetry opt-in test")
TEST.write_text(test_text, encoding="utf-8")

Path(__file__).unlink(missing_ok=True)
