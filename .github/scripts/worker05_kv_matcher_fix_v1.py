from __future__ import annotations

from pathlib import Path

path = Path(__file__).with_name("worker05_kv_canonical_v1.py")
text = path.read_text(encoding="utf-8")
old = '''# Probe loops intentionally isolate one hardware/runtime candidate from the next. Those
# boundaries must catch arbitrary backend failures, but make that policy explicit to Ruff.
replace_expected(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    "        except Exception as exc:\\n",
    "        except Exception as exc:  # noqa: BLE001 - isolate optional benchmark candidate failures\\n",
    expected=1,
)
replace_expected(
    "minecraft_mod_ai/llama_decode_speed_contract.py",
    "            except Exception as exc:\\n",
    "            except Exception as exc:  # noqa: BLE001 - isolate optional KV candidate failures\\n",
    expected=1,
)
'''
new = '''# Probe loops intentionally isolate one hardware/runtime candidate from the next. Mark
# those boundaries by function scope so unrelated broad-exception boundaries cannot match.
def annotate_probe_boundary(function_name: str, comment: str) -> None:
    target = ROOT / "minecraft_mod_ai/llama_decode_speed_contract.py"
    source = target.read_text(encoding="utf-8")
    start = source.find(f"def {function_name}(")
    if start < 0:
        raise SystemExit(f"llama_decode_speed_contract.py: missing {function_name}")
    end = source.find("\\ndef ", start + 1)
    if end < 0:
        end = len(source)
    section = source[start:end]
    old_line = "except Exception as exc:"
    if section.count(old_line) != 1:
        raise SystemExit(
            f"llama_decode_speed_contract.py: {function_name} expected one broad boundary, "
            f"found {section.count(old_line)}"
        )
    section = section.replace(old_line, f"except Exception as exc:  # noqa: BLE001 - {comment}", 1)
    target.write_text(source[:start] + section + source[end:], encoding="utf-8")


annotate_probe_boundary("_probe_kv_types", "isolate optional KV candidate failures")
annotate_probe_boundary("_probe_p_min", "isolate optional benchmark candidate failures")
'''
if text.count(old) != 1:
    raise SystemExit(f"worker05_kv_canonical_v1.py: matcher block count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
Path(__file__).unlink(missing_ok=True)
