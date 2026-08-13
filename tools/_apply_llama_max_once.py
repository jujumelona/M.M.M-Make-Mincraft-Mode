from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(".")
LATEST = "f65e568fd83712c92babbb096b57e572af0ec357"
BUNDLE_SCHEMA = "mmm/native-llama-cuda-bundle-v3-max-t4"
BUNDLE_TAG = "native-llama-f65e568-cuda12.4-max-v3"
BUNDLE_PREFIX = "llama-f65e568-cuda12.4-max"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one occurrence, found {count}")
    return text.replace(old, new, 1)


# 1) Colab runtime: latest upstream source, new setup generation, CUB 3.2 dot path.
path = "tools/colab_runtime_setup.py"
text = read(path)
text = text.replace(
    'SETUP_API_VERSION = "mmm/colab-runtime-setup-v3-prebuilt-cache"',
    'SETUP_API_VERSION = "mmm/colab-runtime-setup-v4-max-native"',
)
text = re.sub(
    r'LLAMA_SERVER_SOURCE_REF = "[0-9a-f]{40}"',
    f'LLAMA_SERVER_SOURCE_REF = "{LATEST}"',
    text,
    count=1,
)
needle = '            "-DGGML_CUDA_GRAPHS=ON",\n'
if '-DGGML_CUDA_CUB_3DOT2=ON' not in text:
    text = replace_once(
        text,
        needle,
        needle
        + '            "-DGGML_CUDA_CUB_3DOT2=ON",\n'
        + '            "-DGGML_CUDA_FA=ON",\n'
        + '            "-DGGML_CUDA_FA_ALL_QUANTS=OFF",\n'
        + '            "-DGGML_LTO=ON",\n',
        label="colab CUDA flags",
    )
write(path, text)

# 2) Native bundle worker: latest upstream + fresh cache generation + max CUDA build.
path = ".github/workflows/build-native-llama-cuda.yml"
text = read(path)
text = re.sub(r"LLAMA_SOURCE_REF: [0-9a-f]{40}", f"LLAMA_SOURCE_REF: {LATEST}", text, count=1)
text = re.sub(r"BUNDLE_SCHEMA: \S+", f"BUNDLE_SCHEMA: {BUNDLE_SCHEMA}", text, count=1)
text = re.sub(r"BUNDLE_PREFIX: \S+", f"BUNDLE_PREFIX: {BUNDLE_PREFIX}", text, count=1)
text = re.sub(r"RELEASE_TAG: \S+", f"RELEASE_TAG: {BUNDLE_TAG}", text, count=1)
if "-DGGML_CUDA_CUB_3DOT2=ON" not in text:
    text = replace_once(
        text,
        "            -DGGML_CUDA_GRAPHS=ON \\\n",
        "            -DGGML_CUDA_GRAPHS=ON \\\n"
        "            -DGGML_CUDA_CUB_3DOT2=ON \\\n"
        "            -DGGML_CUDA_FA=ON \\\n"
        "            -DGGML_CUDA_FA_ALL_QUANTS=OFF \\\n"
        "            -DGGML_LTO=ON \\\n",
        label="workflow CUDA flags",
    )
if "GGML_CUDA_CUB_3DOT2:BOOL=ON" not in text:
    text = replace_once(
        text,
        "          grep -q '^GGML_CUDA_GRAPHS:BOOL=ON$' llama.cpp/build/CMakeCache.txt\n",
        "          grep -q '^GGML_CUDA_GRAPHS:BOOL=ON$' llama.cpp/build/CMakeCache.txt\n"
        "          grep -q '^GGML_CUDA_CUB_3DOT2:BOOL=ON$' llama.cpp/build/CMakeCache.txt\n"
        "          grep -q '^GGML_CUDA_FA:BOOL=ON$' llama.cpp/build/CMakeCache.txt\n"
        "          grep -q '^GGML_LTO:BOOL=ON$' llama.cpp/build/CMakeCache.txt\n",
        label="workflow verification",
    )
if '"cuda_cub_3dot2": True' not in text:
    text = replace_once(
        text,
        '              "cuda_graphs": True,\n',
        '              "cuda_graphs": True,\n'
        '              "cuda_cub_3dot2": True,\n'
        '              "cuda_fa": True,\n'
        '              "lto": True,\n',
        label="bundle manifest features",
    )
text = re.sub(
    r'--title "Native llama\.cpp CUDA [^"]+"',
    '--title "Native llama.cpp latest CUDA max-throughput"',
    text,
    count=1,
)
text = re.sub(
    r'--notes "Verified native llama-server CUDA 12\.4 bundles[^"]+"',
    f'--notes "Verified native llama-server CUDA 12.4 max-throughput bundles. Source: ggml-org/llama.cpp {LATEST}. CUDA Graphs + CUB 3.2 dot path + FlashAttention + LTO."',
    text,
    count=1,
)
write(path, text)

# 3) Loader: new release/cache directory and manifest must prove all build features.
path = "tools/native_llama_bundle.py"
text = read(path)
text = re.sub(
    r'BUNDLE_SCHEMA_VERSION = "[^"]+"',
    f'BUNDLE_SCHEMA_VERSION = "{BUNDLE_SCHEMA}"',
    text,
    count=1,
)
text = re.sub(
    r'BUNDLE_RELEASE_TAG = "[^"]+"',
    f'BUNDLE_RELEASE_TAG = "{BUNDLE_TAG}"',
    text,
    count=1,
)
text = re.sub(
    r'BUNDLE_NAME_PREFIX = "[^"]+"',
    f'BUNDLE_NAME_PREFIX = "{BUNDLE_PREFIX}"',
    text,
    count=1,
)
feature_guard = '''    if manifest.get("cuda_graphs") is not True:
        raise RuntimeError(
            "prebuilt native llama manifest requires cuda_graphs=true"
        )
'''
replacement_guard = feature_guard + '''    if manifest.get("cuda_cub_3dot2") is not True:
        raise RuntimeError(
            "prebuilt native llama manifest requires cuda_cub_3dot2=true"
        )
    if manifest.get("cuda_fa") is not True:
        raise RuntimeError(
            "prebuilt native llama manifest requires cuda_fa=true"
        )
    if manifest.get("lto") is not True:
        raise RuntimeError(
            "prebuilt native llama manifest requires lto=true"
        )
'''
if "requires cuda_cub_3dot2=true" not in text:
    text = replace_once(text, feature_guard, replacement_guard, label="loader feature guard")
write(path, text)

# 4) Existing deterministic T4 probe: allow synthetic context padding for KV buckets.
path = "minecraft_mod_ai/qwen35_t4_single_stream_tuning.py"
text = read(path)
if "def _benchmark_padding()" not in text:
    helper = r'''
def _benchmark_padding() -> str:
    raw = os.environ.get("MMM_QWEN35_T4_BENCH_PAD_CHARS", "").strip()
    try:
        chars = max(0, min(160000, int(raw))) if raw else 0
    except ValueError:
        chars = 0
    if chars <= 0:
        return ""
    pattern = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
    repeats = (chars // len(pattern)) + 1
    filler = (pattern * repeats)[:chars]
    return (
        "Ignore the following calibration-only context. Never repeat it in the answer.\n"
        + filler
        + "\nCalibration context ends here. Perform the exact copy task below.\n"
    )


'''
    text = replace_once(
        text,
        "def _benchmark_request() -> Any:\n",
        helper + "def _benchmark_request() -> Any:\n",
        label="T4 benchmark padding helper",
    )
if '_benchmark_padding() + "Copy this payload exactly' not in text:
    text = replace_once(
        text,
        '                "content": "Copy this payload exactly:\\n" + _EXPECTED_TEXT,\n',
        '                "content": _benchmark_padding() + "Copy this payload exactly:\\n" + _EXPECTED_TEXT,\n',
        label="T4 benchmark padded request",
    )
write(path, text)

# 5) Add adaptive max-speed owner.
MAX_MODULE = 'from __future__ import annotations\n\n"""Adaptive maximum-throughput policy for Qwen3.5-9B-MTP on Tesla T4."""\n\nimport hashlib\nimport json\nimport os\nimport threading\nfrom dataclasses import asdict\nfrom functools import wraps\nfrom pathlib import Path\nfrom typing import Any\n\nfrom .qwen35_mtp_hotpath_contract import (\n    _is_qwen35_mtp,\n    _reclaim_prior_mmm_server,\n)\nfrom . import qwen35_t4_single_stream_tuning as t4\n\n_SCHEMA_VERSION = "mmm/qwen35-t4-max-speed-v3"\n_MARKER = "_mmm_qwen35_t4_max_speed_v3"\n_LOCK = threading.RLock()\n\n_DEFAULT_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16)\n_DEFAULT_P_MINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)\n_DEFAULT_UBATCHES = (512, 1024, 2048)\n_DEFAULT_KV = ("native-default", "q8_0", "q4_0")\n_DEFAULT_BUCKETS = (2048, 8192, 16384, 28672)\n_KV_ENV = "MMM_QWEN35_T4_KV_OVERRIDE"\n_PAD_ENV = "MMM_QWEN35_T4_BENCH_PAD_CHARS"\n_ACTIVE_BUCKET_ENV = "MMM_LLAMA_ACTIVE_KV_BUCKET"\n\n\ndef _enabled() -> bool:\n    return os.environ.get("MMM_QWEN35_T4_MAX_SPEED", "1").strip().lower() not in {\n        "0", "false", "no", "off",\n    }\n\n\ndef _parse_ints(name: str, defaults: tuple[int, ...], *, minimum: int, maximum: int) -> tuple[int, ...]:\n    raw = os.environ.get(name, ",".join(str(value) for value in defaults))\n    values: list[int] = []\n    for token in raw.split(","):\n        try:\n            value = int(token.strip())\n        except ValueError:\n            continue\n        if minimum <= value <= maximum and value not in values:\n            values.append(value)\n    return tuple(values or defaults)\n\n\ndef _widths() -> tuple[int, ...]:\n    return _parse_ints(\n        "MMM_QWEN35_T4_MAX_WIDTHS",\n        _DEFAULT_WIDTHS,\n        minimum=1,\n        maximum=32,\n    )\n\n\ndef _ubatches(batch: int) -> tuple[int, ...]:\n    values = _parse_ints(\n        "MMM_QWEN35_T4_UBATCHES",\n        _DEFAULT_UBATCHES,\n        minimum=64,\n        maximum=max(64, batch),\n    )\n    return tuple(value for value in values if value <= batch) or (min(batch, 512),)\n\n\ndef _p_mins() -> tuple[float, ...]:\n    raw = os.environ.get(\n        "MMM_QWEN35_T4_MAX_P_MIN",\n        ",".join(str(value) for value in _DEFAULT_P_MINS),\n    )\n    values: list[float] = []\n    for token in raw.split(","):\n        try:\n            value = round(float(token.strip()), 4)\n        except ValueError:\n            continue\n        if 0.0 <= value < 1.0 and value not in values:\n            values.append(value)\n    if 0.0 not in values:\n        values.insert(0, 0.0)\n    return tuple(values or _DEFAULT_P_MINS)\n\n\ndef _minimum_gain() -> float:\n    try:\n        return max(1.0, float(os.environ.get("MMM_QWEN35_T4_MAX_MIN_GAIN", "1.005")))\n    except ValueError:\n        return 1.005\n\n\ndef _kv_candidates() -> tuple[str, ...]:\n    aliases = {\n        "native": "native-default",\n        "native-default": "native-default",\n        "f16": "f16",\n        "q8": "q8_0",\n        "q8_0": "q8_0",\n        "q4": "q4_0",\n        "q4_0": "q4_0",\n    }\n    raw = os.environ.get("MMM_QWEN35_T4_MAX_KV", ",".join(_DEFAULT_KV))\n    values: list[str] = []\n    for token in raw.split(","):\n        value = aliases.get(token.strip().lower(), "")\n        if value and value not in values:\n            values.append(value)\n    if "native-default" not in values:\n        values.insert(0, "native-default")\n    return tuple(values or _DEFAULT_KV)\n\n\ndef _kv_mode(config: Any) -> str:\n    aliases = {\n        "auto": "auto",\n        "native": "native-default",\n        "native-default": "native-default",\n        "f16": "f16",\n        "q8": "q8_0",\n        "q8_0": "q8_0",\n        "q4": "q4_0",\n        "q4_0": "q4_0",\n    }\n    raw = os.environ.get("MMM_QWEN35_T4_KV_MODE", "").strip().lower()\n    if raw:\n        return aliases.get(raw, "auto")\n    extra = getattr(config, "extra", {})\n    if isinstance(extra, dict) and extra.get("kv_cache_autotune") is False:\n        return aliases.get(str(extra.get("kv_cache_quant", "q4_0")).strip().lower(), "q4_0")\n    return "auto"\n\n\ndef _context_buckets(config: Any) -> tuple[int, ...]:\n    max_context = max(2048, int(getattr(config, "max_context", 32768) or 32768))\n    values = _parse_ints(\n        "MMM_QWEN35_T4_KV_CONTEXT_BUCKETS",\n        _DEFAULT_BUCKETS,\n        minimum=512,\n        maximum=max_context,\n    )\n    headroom = min(1536, max_context // 4)\n    cap = max(512, max_context - headroom)\n    clipped = sorted({min(value, cap) for value in values})\n    return tuple(value for value in clipped if value >= 512) or (min(2048, cap),)\n\n\ndef _request_tokens(request: Any) -> int:\n    chars = 0\n    for message in getattr(request, "messages", ()) or ():\n        if not isinstance(message, dict):\n            continue\n        value = message.get("content", "")\n        if isinstance(value, str):\n            chars += len(value)\n        else:\n            try:\n                chars += len(json.dumps(value, ensure_ascii=False))\n            except Exception:\n                pass\n    return max(1, chars // 3)\n\n\ndef _bucket_for_request(config: Any, request: Any) -> int:\n    estimate = _request_tokens(request)\n    buckets = _context_buckets(config)\n    for bucket in buckets:\n        if estimate <= bucket:\n            return bucket\n    return buckets[-1]\n\n\ndef _core_cache_path() -> Path:\n    explicit = os.environ.get("MMM_QWEN35_T4_MAX_CACHE", "").strip()\n    if explicit:\n        return Path(explicit).expanduser().resolve()\n    return (Path.home() / ".cache" / "mmm" / "qwen35-t4-max-speed-v3.json").resolve()\n\n\ndef _kv_cache_path() -> Path:\n    explicit = os.environ.get("MMM_QWEN35_T4_KV_BUCKET_CACHE", "").strip()\n    if explicit:\n        return Path(explicit).expanduser().resolve()\n    return (Path.home() / ".cache" / "mmm" / "qwen35-t4-kv-buckets-v3.json").resolve()\n\n\ndef _fingerprint(autotune: Any, config: Any, binary: str, model_path: str) -> str:\n    batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)\n    payload = {\n        "schema": _SCHEMA_VERSION,\n        "base": autotune._fingerprint(config, binary, model_path),\n        "hardware": t4._hardware_identity(autotune),\n        "server": autotune._server_version(binary),\n        "widths": list(_widths()),\n        "p_mins": list(_p_mins()),\n        "ubatches": list(_ubatches(batch)),\n        "kv": list(_kv_candidates()),\n        "buckets": list(_context_buckets(config)),\n        "probe_tokens": t4._probe_tokens(autotune, config),\n        "benchmark": t4._EXPECTED_DIGEST,\n    }\n    return hashlib.sha256(\n        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()\n    ).hexdigest()\n\n\ndef _tps(probe: Any) -> float:\n    return float(getattr(probe, "predicted_tps", 0.0) or 0.0)\n\n\ndef _valid(probe: Any, baseline: Any) -> bool:\n    return t4._valid(probe, baseline)\n\n\ndef _best(baseline: Any, probes: list[Any]) -> Any:\n    eligible = [baseline] + [\n        probe for probe in probes if probe is not baseline and _valid(probe, baseline)\n    ]\n    winner = max(eligible, key=_tps)\n    if winner is not baseline and _tps(winner) < _tps(baseline) * _minimum_gain():\n        return baseline\n    return winner\n\n\ndef _variant(\n    autotune: Any,\n    *,\n    name: str,\n    ubatch: int,\n    width: int = 0,\n    p_min: float = 0.0,\n) -> Any:\n    return t4._variant(\n        autotune,\n        name=name,\n        ubatch=ubatch,\n        width=width,\n        p_min=p_min,\n    )\n\n\ndef _log(probe: Any, stage: str, *, bucket: int | None = None) -> None:\n    detail = (\n        f"{_tps(probe):.2f} tok/s"\n        if bool(getattr(probe, "ok", False))\n        else str(getattr(probe, "error", "failed"))\n    )\n    suffix = f" bucket={bucket}" if bucket is not None else ""\n    print(\n        "llama server: T4 max-speed probe",\n        f"stage={stage}",\n        probe.variant.name,\n        f"kv={getattr(probe, \'kv\', \'native-default\')}{suffix}",\n        f"-> {detail}",\n        flush=True,\n    )\n\n\ndef _probe(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n    candidate: Any,\n    *,\n    kv: str = "native-default",\n    pad_chars: int = 0,\n) -> Any:\n    old_pad = os.environ.get(_PAD_ENV)\n    if pad_chars > 0:\n        os.environ[_PAD_ENV] = str(pad_chars)\n    else:\n        os.environ.pop(_PAD_ENV, None)\n    try:\n        return t4._probe(\n            autotune,\n            binary,\n            model_path,\n            config,\n            candidate,\n            kv=kv,\n        )\n    finally:\n        if old_pad is None:\n            os.environ.pop(_PAD_ENV, None)\n        else:\n            os.environ[_PAD_ENV] = old_pad\n\n\ndef _measure_core(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n) -> tuple[Any, float, float]:\n    batch = autotune._env_int("MMM_LLAMA_BATCH", 2048)\n    ubatches = _ubatches(batch)\n    seed_ubatch = 512 if 512 in ubatches else ubatches[0]\n    baseline = _probe(\n        autotune,\n        binary,\n        model_path,\n        config,\n        _variant(autotune, name="qwen35-t4-max-baseline", ubatch=seed_ubatch),\n    )\n    _log(baseline, "baseline")\n    if not bool(getattr(baseline, "ok", False)) or _tps(baseline) <= 0:\n        raise RuntimeError("T4 max-speed baseline probe failed")\n\n    width_probes: list[Any] = []\n    all_probes: list[Any] = [baseline]\n    for width in _widths():\n        probe = _probe(\n            autotune,\n            binary,\n            model_path,\n            config,\n            _variant(\n                autotune,\n                name=f"qwen35-t4-max-mtp-{width}",\n                ubatch=seed_ubatch,\n                width=width,\n            ),\n        )\n        _log(probe, "mtp-width")\n        all_probes.append(probe)\n        if _valid(probe, baseline):\n            width_probes.append(probe)\n\n    seeds = sorted(width_probes, key=_tps, reverse=True)[:2]\n    widest = next(\n        (\n            probe\n            for probe in width_probes\n            if int(getattr(probe.variant, "draft_n_max", 0) or 0) == max(_widths())\n        ),\n        None,\n    )\n    if widest is not None and widest not in seeds:\n        seeds.append(widest)\n\n    for seed in seeds:\n        width = int(getattr(seed.variant, "draft_n_max", 0) or 0)\n        for p_min in _p_mins():\n            if p_min == 0.0:\n                continue\n            probe = _probe(\n                autotune,\n                binary,\n                model_path,\n                config,\n                _variant(\n                    autotune,\n                    name=f"qwen35-t4-max-mtp-{width}|pm{p_min:g}",\n                    ubatch=seed_ubatch,\n                    width=width,\n                    p_min=p_min,\n                ),\n            )\n            _log(probe, "p-min")\n            all_probes.append(probe)\n\n    selected_probe = _best(baseline, all_probes)\n    selected = selected_probe.variant\n\n    ubatch_probes = [selected_probe]\n    for ubatch in ubatches:\n        if ubatch == int(getattr(selected, "ubatch", seed_ubatch) or seed_ubatch):\n            continue\n        probe = _probe(\n            autotune,\n            binary,\n            model_path,\n            config,\n            _variant(\n                autotune,\n                name=f"{selected.name}|ub{ubatch}",\n                ubatch=ubatch,\n                width=int(getattr(selected, "draft_n_max", 0) or 0),\n                p_min=float(getattr(selected, "draft_p_min", 0.0) or 0.0),\n            ),\n        )\n        _log(probe, "ubatch")\n        if _valid(probe, baseline):\n            ubatch_probes.append(probe)\n    selected_probe = max(ubatch_probes, key=_tps)\n    selected = selected_probe.variant\n    return selected, _tps(baseline), _tps(selected_probe)\n\n\ndef _load_core(autotune: Any, fingerprint: str) -> tuple[Any, float, float] | None:\n    try:\n        payload = json.loads(_core_cache_path().read_text(encoding="utf-8"))\n    except Exception:\n        return None\n    if payload.get("schema") != _SCHEMA_VERSION or payload.get("fingerprint") != fingerprint:\n        return None\n    try:\n        return (\n            autotune.ServerVariant(**payload["selected"]),\n            float(payload["baseline_tps"]),\n            float(payload["selected_tps"]),\n        )\n    except Exception:\n        return None\n\n\ndef _save_core(fingerprint: str, selected: Any, baseline_tps: float, selected_tps: float) -> None:\n    path = _core_cache_path()\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_suffix(path.suffix + ".tmp")\n    temporary.write_text(\n        json.dumps(\n            {\n                "schema": _SCHEMA_VERSION,\n                "fingerprint": fingerprint,\n                "selected": asdict(selected),\n                "baseline_tps": baseline_tps,\n                "selected_tps": selected_tps,\n            },\n            indent=2,\n            sort_keys=True,\n        )\n        + "\\n",\n        encoding="utf-8",\n    )\n    os.replace(temporary, path)\n\n\ndef _load_kv_map(fingerprint: str) -> dict[str, str]:\n    try:\n        payload = json.loads(_kv_cache_path().read_text(encoding="utf-8"))\n    except Exception:\n        return {}\n    if payload.get("schema") != _SCHEMA_VERSION or payload.get("fingerprint") != fingerprint:\n        return {}\n    raw = payload.get("buckets", {})\n    if not isinstance(raw, dict):\n        return {}\n    allowed = set(_kv_candidates())\n    return {\n        str(key): str(value)\n        for key, value in raw.items()\n        if str(value) in allowed\n    }\n\n\ndef _save_kv_map(fingerprint: str, values: dict[str, str]) -> None:\n    path = _kv_cache_path()\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_suffix(path.suffix + ".tmp")\n    temporary.write_text(\n        json.dumps(\n            {\n                "schema": _SCHEMA_VERSION,\n                "fingerprint": fingerprint,\n                "buckets": dict(sorted(values.items(), key=lambda item: int(item[0]))),\n            },\n            indent=2,\n            sort_keys=True,\n        )\n        + "\\n",\n        encoding="utf-8",\n    )\n    os.replace(temporary, path)\n\n\ndef _measure_kv_bucket(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n    selected: Any,\n    bucket: int,\n) -> str:\n    pad_chars = max(0, int(bucket * 4.0) - 4096)\n    baseline = _probe(\n        autotune,\n        binary,\n        model_path,\n        config,\n        selected,\n        kv="native-default",\n        pad_chars=pad_chars,\n    )\n    _log(baseline, "kv", bucket=bucket)\n    if not bool(getattr(baseline, "ok", False)) or _tps(baseline) <= 0:\n        raise RuntimeError(f"KV baseline failed for context bucket {bucket}")\n\n    candidates: list[Any] = [baseline]\n    for kv in _kv_candidates():\n        if kv == "native-default":\n            continue\n        probe = _probe(\n            autotune,\n            binary,\n            model_path,\n            config,\n            selected,\n            kv=kv,\n            pad_chars=pad_chars,\n        )\n        _log(probe, "kv", bucket=bucket)\n        if _valid(probe, baseline):\n            candidates.append(probe)\n    winner = max(candidates, key=_tps)\n    if winner is not baseline and _tps(winner) < _tps(baseline) * _minimum_gain():\n        return "native-default"\n    return str(getattr(winner, "kv", "native-default"))\n\n\ndef _select_kv(\n    autotune: Any,\n    binary: str,\n    model_path: str,\n    config: Any,\n    request: Any,\n    selected: Any,\n    fingerprint: str,\n) -> tuple[str, int]:\n    bucket = _bucket_for_request(config, request)\n    mode = _kv_mode(config)\n    if mode != "auto":\n        return mode, bucket\n\n    values = _load_kv_map(fingerprint)\n    key = str(bucket)\n    cached = values.get(key)\n    if cached:\n        return cached, bucket\n    selected_kv = _measure_kv_bucket(\n        autotune,\n        binary,\n        model_path,\n        config,\n        selected,\n        bucket,\n    )\n    values[key] = selected_kv\n    _save_kv_map(fingerprint, values)\n    return selected_kv, bucket\n\n\ndef _export(selected: Any, selected_kv: str, bucket: int) -> None:\n    os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = str(selected.spec_type)\n    os.environ["MMM_LLAMA_ACTIVE_DRAFT_N_MAX"] = str(selected.draft_n_max)\n    os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = "1"\n    os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(selected.ubatch)\n    os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = "0"\n    os.environ["MMM_LLAMA_ACTIVE_MTP_P_MIN"] = f"{selected.draft_p_min:g}"\n    os.environ["MMM_LLAMA_ACTIVE_TUNING_OBJECTIVE"] = "single_stream"\n    os.environ["MMM_LLAMA_ACTIVE_KV_CACHE"] = selected_kv\n    os.environ[_ACTIVE_BUCKET_ENV] = str(bucket)\n    if selected_kv == "native-default":\n        os.environ.pop(_KV_ENV, None)\n    else:\n        os.environ[_KV_ENV] = selected_kv\n\n\ndef _shutdown_managed(autotune: Any) -> None:\n    shutdown = getattr(autotune, "_shutdown_managed_server", None)\n    if callable(shutdown):\n        shutdown()\n        return\n    process = getattr(autotune, "_MANAGED_PROCESS", None)\n    autotune._stop_server(process)\n    autotune._MANAGED_PROCESS = None\n    autotune._MANAGED_URL = None\n    os.environ.pop("LLAMA_SERVER_URL", None)\n\n\ndef install(autotune: Any) -> None:\n    current = autotune.ensure_tuned_server\n    if getattr(current, _MARKER, False):\n        return\n\n    @wraps(current)\n    def ensure_max_speed(config: Any, request: Any) -> str:\n        if not (_enabled() and _is_qwen35_mtp(config) and t4._is_t4_runtime(autotune)):\n            return current(config, request)\n\n        with _LOCK, autotune._AUTOTUNE_LOCK:\n            binary = autotune._server_binary()\n            if binary is None:\n                return current(config, request)\n            model_path = autotune._resolve_model_path(config)\n            fingerprint = _fingerprint(autotune, config, binary, model_path)\n\n            core = _load_core(autotune, fingerprint)\n            if core is None:\n                _reclaim_prior_mmm_server()\n                try:\n                    selected, baseline_tps, selected_tps = _measure_core(\n                        autotune, binary, model_path, config\n                    )\n                except Exception as exc:\n                    print(\n                        "llama server: T4 max-speed calibration failed; using conservative T4 tuner",\n                        f"{type(exc).__name__}: {exc}",\n                        flush=True,\n                    )\n                    return current(config, request)\n                _save_core(fingerprint, selected, baseline_tps, selected_tps)\n                source = "measured"\n            else:\n                selected, baseline_tps, selected_tps = core\n                source = "cache"\n\n            try:\n                selected_kv, bucket = _select_kv(\n                    autotune,\n                    binary,\n                    model_path,\n                    config,\n                    request,\n                    selected,\n                    fingerprint,\n                )\n            except Exception as exc:\n                print(\n                    "llama server: context-KV calibration failed; using native KV",\n                    f"{type(exc).__name__}: {exc}",\n                    flush=True,\n                )\n                selected_kv = "native-default"\n                bucket = _bucket_for_request(config, request)\n\n            process = getattr(autotune, "_MANAGED_PROCESS", None)\n            managed_url = str(getattr(autotune, "_MANAGED_URL", "") or "")\n            active_kv = os.environ.get("MMM_LLAMA_ACTIVE_KV_CACHE", "").strip().lower()\n            if process is not None and process.poll() is None and managed_url:\n                if active_kv == selected_kv:\n                    os.environ[_ACTIVE_BUCKET_ENV] = str(bucket)\n                    return managed_url\n                _shutdown_managed(autotune)\n\n            _export(selected, selected_kv, bucket)\n            url = autotune._launch_selected(binary, model_path, config, selected)\n            speedup = selected_tps / baseline_tps if baseline_tps > 0 else 1.0\n            print(\n                "llama server: T4 max-speed production profile",\n                f"source={source}",\n                f"spec={selected.spec_type}",\n                f"n_max={selected.draft_n_max}",\n                f"p_min={selected.draft_p_min:g}",\n                f"ubatch={selected.ubatch}",\n                f"kv={selected_kv}",\n                f"bucket={bucket}",\n                f"baseline={baseline_tps:.2f}",\n                f"selected={selected_tps:.2f}",\n                f"speedup={speedup:.3f}x",\n                flush=True,\n            )\n            return url\n\n    setattr(ensure_max_speed, _MARKER, True)\n    ensure_max_speed._mmm_qwen35_t4_max_speed = True\n    autotune.ensure_tuned_server = ensure_max_speed\n\n\n__all__ = [\n    "_bucket_for_request",\n    "_context_buckets",\n    "_kv_candidates",\n    "_kv_mode",\n    "_p_mins",\n    "_ubatches",\n    "_widths",\n    "install",\n]\n'
write("minecraft_mod_ai/qwen35_t4_max_speed_tuning.py", MAX_MODULE)

# 6) Compose the max-speed owner after conservative T4 tuning.
path = "minecraft_mod_ai/llama_tuning_pipeline.py"
text = read(path)
text = re.sub(r"_TUNING_PIPELINE_VERSION = \d+", "_TUNING_PIPELINE_VERSION = 8", text, count=1)
if "qwen35_t4_max_speed_tuning" not in text:
    import_block = '''        from .qwen35_t4_single_stream_tuning import (
            install as install_qwen35_t4_single_stream,
        )
'''
    text = replace_once(
        text,
        import_block,
        import_block
        + "        from .qwen35_t4_max_speed_tuning import install as install_qwen35_t4_max_speed\n",
        label="pipeline max-speed import",
    )
    text = replace_once(
        text,
        "            install_qwen35_t4_single_stream(self.autotune)\n",
        "            install_qwen35_t4_single_stream(self.autotune)\n"
        "            install_qwen35_t4_max_speed(self.autotune)\n",
        label="pipeline max-speed install",
    )
write(path, text)

# 7) Structural composer test: max-speed is an owned tuning stage.
path = "tests/test_runtime_bootstrap_clean.py"
text = read(path)
if '"qwen35_t4_max_speed_tuning",' not in text:
    text = replace_once(
        text,
        '        "qwen35_t4_single_stream_tuning",\n',
        '        "qwen35_t4_single_stream_tuning",\n'
        '        "qwen35_t4_max_speed_tuning",\n',
        label="composer expected module",
    )
if 'source.index("install_qwen35_t4_max_speed(")' not in text:
    text = replace_once(
        text,
        '        < source.index("install_qwen35_t4_single_stream(")\n'
        '        < source.index("install_single_stream_agentic_policy(")\n',
        '        < source.index("install_qwen35_t4_single_stream(")\n'
        '        < source.index("install_qwen35_t4_max_speed(")\n'
        '        < source.index("install_single_stream_agentic_policy(")\n',
        label="composer ordering",
    )
write(path, text)

# 8) Native bundle contract tests updated to the new cache generation/features.
path = "tests/test_native_llama_prebuilt_contract.py"
text = read(path)
text = text.replace(
    "def test_emergency_source_build_enables_cuda_graphs() -> None:",
    "def test_emergency_source_build_enables_max_cuda_path() -> None:",
)
if 'assert \'"-DGGML_CUDA_CUB_3DOT2=ON"\' in fallback' not in text:
    text = replace_once(
        text,
        '    assert \'"-DGGML_CUDA_GRAPHS=ON"\' in fallback\n',
        '    assert \'"-DGGML_CUDA_GRAPHS=ON"\' in fallback\n'
        '    assert \'"-DGGML_CUDA_CUB_3DOT2=ON"\' in fallback\n'
        '    assert \'"-DGGML_CUDA_FA=ON"\' in fallback\n'
        '    assert \'"-DGGML_LTO=ON"\' in fallback\n',
        label="source build tests",
    )
text = re.sub(
    r'assert "BUNDLE_SCHEMA: [^"]+" in worker',
    f'assert "BUNDLE_SCHEMA: {BUNDLE_SCHEMA}" in worker',
    text,
    count=1,
)
text = re.sub(
    r'assert "RELEASE_TAG: [^"]+" in worker',
    f'assert "RELEASE_TAG: {BUNDLE_TAG}" in worker',
    text,
    count=1,
)
if 'assert "-DGGML_CUDA_CUB_3DOT2=ON" in worker' not in text:
    text = replace_once(
        text,
        '    assert "-DGGML_CUDA_GRAPHS=ON" in worker\n',
        '    assert "-DGGML_CUDA_GRAPHS=ON" in worker\n'
        '    assert "-DGGML_CUDA_CUB_3DOT2=ON" in worker\n'
        '    assert "-DGGML_CUDA_FA=ON" in worker\n'
        '    assert "-DGGML_LTO=ON" in worker\n',
        label="workflow feature tests",
    )
text = re.sub(
    r'assert helper\.BUNDLE_SCHEMA_VERSION == "[^"]+"',
    f'assert helper.BUNDLE_SCHEMA_VERSION == "{BUNDLE_SCHEMA}"',
    text,
    count=1,
)
text = re.sub(
    r'assert helper\.BUNDLE_RELEASE_TAG == "[^"]+"',
    f'assert helper.BUNDLE_RELEASE_TAG == "{BUNDLE_TAG}"',
    text,
    count=1,
)
needle = '''                "cuda_graphs": False,
                "files": {},
'''
if needle in text:
    text = text.replace(
        needle,
        '''                "cuda_graphs": False,
                "cuda_cub_3dot2": True,
                "cuda_fa": True,
                "lto": True,
                "files": {},
''',
        1,
    )
write(path, text)

TEST_MODULE = r'''from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.qwen35_t4_max_speed_tuning import (
    _bucket_for_request,
    _context_buckets,
    _kv_candidates,
    _kv_mode,
    _p_mins,
    _ubatches,
    _widths,
)


def test_max_width_search_reaches_sixteen(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_MAX_WIDTHS", raising=False)
    assert _widths() == (1, 2, 3, 4, 6, 8, 12, 16)


def test_max_p_min_search_contains_confidence_region(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_MAX_P_MIN", raising=False)
    assert _p_mins() == (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def test_ubatch_search_covers_t4_decode_shapes(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_UBATCHES", raising=False)
    assert _ubatches(2048) == (512, 1024, 2048)


def test_kv_auto_and_manual_modes(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_MAX_KV", raising=False)
    assert _kv_candidates() == ("native-default", "q8_0", "q4_0")
    config = SimpleNamespace(extra={})
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "q8")
    assert _kv_mode(config) == "q8_0"
    monkeypatch.setenv("MMM_QWEN35_T4_KV_MODE", "auto")
    assert _kv_mode(config) == "auto"


def test_context_bucket_selection_tracks_request_size(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_T4_KV_CONTEXT_BUCKETS", raising=False)
    config = SimpleNamespace(max_context=32768)
    assert _context_buckets(config) == (2048, 8192, 16384, 28672)
    short = SimpleNamespace(messages=({"role": "user", "content": "x" * 900},))
    long = SimpleNamespace(messages=({"role": "user", "content": "x" * 30000},))
    assert _bucket_for_request(config, short) == 2048
    assert _bucket_for_request(config, long) == 16384
'''
write("tests/test_qwen35_t4_max_speed_tuning.py", TEST_MODULE)

# Remove one-shot helpers from the resulting commit.
(ROOT / ".github/workflows/apply-llama-max-once.yml").unlink()
(ROOT / "tools/_apply_llama_max_once.py").unlink()

print("patched latest llama engine + six-axis T4 max-speed policy")
