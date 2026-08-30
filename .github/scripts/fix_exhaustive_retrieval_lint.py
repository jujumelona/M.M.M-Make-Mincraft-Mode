from pathlib import Path

root = Path.cwd()
gh = root / "minecraft_mod_ai" / "github_adaptive_retrieval.py"
rg = root / "minecraft_mod_ai" / "research_grounded_rag_contract.py"
pd = root / "minecraft_mod_ai" / "pre_design_research_pipeline.py"

text = gh.read_text(encoding="utf-8")
old = '''    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):\n        if len(value) >= 2:\n            owner, repo = str(value[0]).strip(), str(value[1]).strip().removesuffix(".git")\n            if owner and repo:\n                return owner, repo\n'''
new = '''    if (\n        isinstance(value, Sequence)\n        and not isinstance(value, (str, bytes, bytearray))\n        and len(value) >= 2\n    ):\n        owner, repo = str(value[0]).strip(), str(value[1]).strip().removesuffix(".git")\n        if owner and repo:\n            return owner, repo\n'''
if old not in text:
    raise SystemExit("expected _repo_tuple block not found")
gh.write_text(text.replace(old, new, 1), encoding="utf-8")

text = rg.read_text(encoding="utf-8")
old = '''    except Exception as exc:\n        return {\n            "status": "build_failed",\n'''
new = '''    except Exception as exc:  # noqa: BLE001 - fail-closed index build boundary\n        return {\n            "status": "build_failed",\n'''
if old not in text:
    raise SystemExit("expected index build exception boundary not found")
rg.write_text(text.replace(old, new, 1), encoding="utf-8")

text = pd.read_text(encoding="utf-8")
old = '''        except Exception as exc:\n            diagnostic = _exception_payload(exc)\n'''
new = '''        except Exception as exc:  # noqa: BLE001 - diagnostic boundary must capture provider failures\n            diagnostic = _exception_payload(exc)\n'''
if old not in text:
    raise SystemExit("expected technology radar exception boundary not found")
pd.write_text(text.replace(old, new, 1), encoding="utf-8")
