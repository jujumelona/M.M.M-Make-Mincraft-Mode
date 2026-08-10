from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


SETUP_API_VERSION = "mmm/colab-runtime-setup-v1"
RECEIPT_SCHEMA_VERSION = "mmm/colab-setup-receipt-v1"
LOCAL_PROFILES = frozenset(
    {
        "t4_quality",
        "t4_local",
        "Qwen3.6-35B_23GB",
        "Qwen3.6-27B_18GB",
        "Qwen3.6-27B_14GB",
        "Qwen3.5-9B_6GB",
        "Gemma4-26B_14GB",
        "Gemma4-12B_7GB",
    }
)
SUPPORTED_PROFILES = LOCAL_PROFILES | {"remote_quality"}
REMOTE_TEXT_ROLES = ("PLANNER", "RESEARCH", "CODER", "CODER_SAFE", "VISION")
REMOTE_PROJECT_INSTALL_TARGET = (
    ".[ui,rag,image,speech,production-audio,training]"
)
LOCAL_PROJECT_INSTALL_TARGET = (
    ".[ui,local-model,rag,image,speech,production-audio,training]"
)
QWEN_FASTPATH_REQUIREMENT = (
    "flash-linear-attention[cuda,conv1d]>=0.5.1,<0.6"
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_remote_url(value: str) -> str:
    """Return a receipt-safe endpoint without credentials, query, or fragment."""

    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return ""
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), "", ""))


def _validated_remote_url(value: str) -> str:
    endpoint = value.strip()
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("remote_quality requires a valid HTTPS API base URL.") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError(
            "remote_quality requires an HTTPS API base URL without embedded "
            "credentials, query parameters, or fragments."
        )
    return endpoint


def setup_request_fingerprint(
    *,
    repo_dir: str | Path,
    used_commit: str,
    model_profile: str,
    save_to_google_drive: bool,
    remote_base_url: str = "",
    remote_text_model: str = "",
    remote_image_model: str = "",
    remote_speech_model: str = "",
) -> str:
    """Fingerprint non-secret setup inputs, including the exact remote endpoint."""

    request = {
        "setup_api_version": SETUP_API_VERSION,
        "repo_dir": str(Path(repo_dir).resolve()),
        "used_commit": used_commit.strip(),
        "model_profile": model_profile.strip(),
        "save_to_google_drive": bool(save_to_google_drive),
        "remote_base_url": remote_base_url.strip(),
        "remote_text_model": remote_text_model.strip(),
        "remote_image_model": remote_image_model.strip(),
        "remote_speech_model": remote_speech_model.strip(),
    }
    return hashlib.sha256(_canonical_json(request).encode("utf-8")).hexdigest()


def _git_head(repo_dir: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _tracked_changes(repo_dir: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_dir),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_loaded_engine_origin(repo_dir: Path) -> None:
    loaded_module = sys.modules.get("minecraft_mod_ai")
    if loaded_module is None:
        return
    module_file = getattr(loaded_module, "__file__", "") or ""
    package_root = (repo_dir / "minecraft_mod_ai").resolve()
    if not module_file or not Path(module_file).resolve().is_relative_to(package_root):
        raise RuntimeError(
            "minecraft_mod_ai is loaded from a different checkout. "
            "restart the Colab runtime and rerun from cell 1."
        )


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _validate_checkout(
    *,
    repo_dir: Path,
    used_commit: str,
    previous_commit: str,
    engine_was_loaded: bool,
    engine_module_file: str,
) -> None:
    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Not a Git checkout: {repo_dir}")
    actual_commit = _git_head(repo_dir)
    if actual_commit != used_commit:
        raise RuntimeError(
            "The pulled checkout changed before setup started: "
            f"expected {used_commit}, found {actual_commit}."
        )
    if _tracked_changes(repo_dir):
        raise RuntimeError(
            "The pulled checkout contains tracked local changes. Remove the "
            "Colab checkout and rerun setup cell 2."
        )
    if engine_was_loaded and (
        not previous_commit or previous_commit.strip() != used_commit
    ):
        print(
            f"🔄 Automatically reloading engine modules (commit {previous_commit[:7] if previous_commit else 'old'} -> {used_commit[:7]})...",
            flush=True,
        )
        to_purge = [
            name
            for name in list(sys.modules.keys())
            if name == "minecraft_mod_ai" or name.startswith("minecraft_mod_ai.")
        ]
        for name in to_purge:
            sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _require_local_cuda() -> Any:
    try:
        import torch
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch or packaging is not installed. Select a Colab GPU runtime."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The selected local profile requires a Colab GPU runtime; CUDA is "
            "unavailable."
        )
    return torch





def _install_llama_cpp() -> None:
    """Install pre-built llama-cpp-python CUDA wheel in seconds, avoiding source compilation."""
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_supports_gpu") and llama_cpp.llama_supports_gpu():
            print("✅ Pre-built llama-cpp-python CUDA wheel with GPU support verified.", flush=True)
            return
    except Exception:
        pass

    print("⚡ Installing pre-built llama-cpp-python CUDA wheel...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "llama-cpp-python",
            "--extra-index-url",
            "https://abetlen.github.io/llama-cpp-python/whl/cu124",
            "--no-cache-dir",
        ],
        check=False,
    )


def _install_project(*, local_profile: bool) -> None:
    if local_profile:
        _install_llama_cpp()
    target = (
        LOCAL_PROJECT_INSTALL_TARGET
        if local_profile
        else REMOTE_PROJECT_INSTALL_TARGET
    )
    print(f"📦 Installing M.M.M project dependencies ({target})...", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-build-isolation",
        "-e",
        target,
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
    retcode = process.wait()
    if retcode != 0:
        raise subprocess.CalledProcessError(retcode, cmd)
    print("✅ M.M.M project package installed successfully!", flush=True)


def _verify_qwen_fastpath(*, torch: Any, transformers_was_loaded: bool) -> None:
    importlib.invalidate_caches()
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
        from transformers.models.qwen3_5 import modeling_qwen3_5
        if not getattr(modeling_qwen3_5, "is_fast_path_available", False):
            print("ℹ️ Qwen3.5 fast path kernels not active; standard PyTorch execution path enabled.", flush=True)
            return
        print("✅ Qwen3.5 CUDA fast-path kernels verified.", flush=True)
    except Exception:
        print("ℹ️ Standard PyTorch execution path silently falling back.", flush=True)
        return

    try:
        with torch.inference_mode():
            dtype = torch.float16
            conv_x = torch.randn((1, 8, 16), device="cuda", dtype=dtype)
            conv_weight = torch.randn((8, 4), device="cuda", dtype=dtype)
            conv_out = causal_conv1d_fn(
                conv_x, conv_weight, activation="silu"
            )
            conv_state = torch.zeros((1, 8, 4), device="cuda", dtype=dtype)
            conv_step = causal_conv1d_update(
                conv_x[:, :, -1], conv_state, conv_weight, activation="silu"
            )

            q = torch.randn((1, 16, 1, 16), device="cuda", dtype=dtype)
            k = torch.nn.functional.normalize(
                torch.randn((1, 16, 1, 16), device="cuda", dtype=torch.float32),
                dim=-1,
            ).to(dtype)
            v = torch.randn((1, 16, 1, 16), device="cuda", dtype=dtype)
            g = torch.nn.functional.logsigmoid(
                torch.randn((1, 16, 1), device="cuda", dtype=torch.float32)
            )
            beta = torch.sigmoid(
                torch.randn((1, 16, 1), device="cuda", dtype=dtype)
            )
            chunk_out, recurrent_state = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            recurrent_out, next_recurrent_state = fused_recurrent_gated_delta_rule(
                q[:, :1],
                k[:, :1],
                v[:, :1],
                g=g[:, :1],
                beta=beta[:, :1],
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            smoke_outputs = (
                conv_out,
                conv_step,
                chunk_out,
                recurrent_state,
                recurrent_out,
                next_recurrent_state,
            )
            if any(not torch.isfinite(output).all() for output in smoke_outputs):
                raise RuntimeError("a Qwen3.5 fast kernel returned non-finite values")
            torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(
            "Qwen3.5 CUDA fast-kernel smoke test failed. Setup will not continue "
            "with an unverified slow fallback."
        ) from exc

    print(
        "Qwen3.5 fast kernels:",
        f"flash-linear-attention={package_version('flash-linear-attention')}",
        f"causal-conv1d={package_version('causal-conv1d')}",
    )


def _configure_output(save_to_google_drive: bool) -> str:
    if save_to_google_drive:
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError(
                "Google Drive output was requested outside Google Colab."
            ) from exc
        drive.mount("/content/drive")
        return "/content/drive/MyDrive/M.M.M-output"
    return "/content/mmm-output"


def _configure_remote(
    *,
    remote_base_url: str,
    remote_text_model: str,
    remote_image_model: str,
    remote_speech_model: str,
) -> None:
    from getpass import getpass

    endpoint = _validated_remote_url(remote_base_url)
    text_model = remote_text_model.strip()
    if not text_model:
        raise ValueError("remote_quality requires a text model name.")
    remote_key = getpass("Remote model API key: ").strip()
    if not remote_key:
        raise ValueError("The remote model API key is empty.")

    for role in REMOTE_TEXT_ROLES:
        os.environ[f"MMM_{role}_BASE_URL"] = endpoint
        os.environ[f"MMM_{role}_MODEL"] = text_model
        os.environ[f"MMM_{role}_API_KEY"] = remote_key
    os.environ["MMM_IMAGE_BASE_URL"] = endpoint
    os.environ["MMM_IMAGE_MODEL"] = remote_image_model.strip() or text_model
    os.environ["MMM_IMAGE_API_KEY"] = remote_key
    os.environ["MMM_SPEECH_BASE_URL"] = endpoint
    os.environ["MMM_SPEECH_MODEL"] = remote_speech_model.strip() or text_model
    os.environ["MMM_SPEECH_API_KEY"] = remote_key


def _assert_remote_environment(
    *,
    remote_base_url: str,
    remote_text_model: str,
    remote_image_model: str,
    remote_speech_model: str,
) -> None:
    endpoint = _validated_remote_url(remote_base_url)
    text_model = remote_text_model.strip()
    expected: dict[str, str] = {}
    for role in REMOTE_TEXT_ROLES:
        expected[f"MMM_{role}_BASE_URL"] = endpoint
        expected[f"MMM_{role}_MODEL"] = text_model
    expected.update(
        {
            "MMM_IMAGE_BASE_URL": endpoint,
            "MMM_IMAGE_MODEL": remote_image_model.strip() or text_model,
            "MMM_SPEECH_BASE_URL": endpoint,
            "MMM_SPEECH_MODEL": remote_speech_model.strip() or text_model,
        }
    )
    mismatched = [name for name, value in expected.items() if os.environ.get(name) != value]
    api_key_names = [
        *(f"MMM_{role}_API_KEY" for role in REMOTE_TEXT_ROLES),
        "MMM_IMAGE_API_KEY",
        "MMM_SPEECH_API_KEY",
    ]
    missing_keys = [name for name in api_key_names if not os.environ.get(name)]
    if mismatched or missing_keys:
        raise RuntimeError(
            "Remote model environment changed after setup. Rerun setup cell 2 "
            "before planning or building."
        )


def _runtime_details(torch: Any | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
    }
    if details["cuda_available"]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        details.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(0)
                ),
                "vram_free_bytes": int(free_bytes),
                "vram_total_bytes": int(total_bytes),
            }
        )
    return details


def _build_receipt(
    *,
    repo_dir: Path,
    used_commit: str,
    model_profile: str,
    save_to_google_drive: bool,
    output_root: str,
    remote_base_url: str,
    remote_text_model: str,
    remote_image_model: str,
    remote_speech_model: str,
    setup_fingerprint: str,
    torch: Any | None,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "setup_api_version": SETUP_API_VERSION,
        "setup_script_sha256": _file_sha256(script_path),
        "repo_dir": str(repo_dir),
        "used_commit": used_commit,
        "model_profile": model_profile,
        "backend": "local_cuda" if model_profile in LOCAL_PROFILES else "remote_api",
        "save_to_google_drive": bool(save_to_google_drive),
        "output_root": output_root,
        "process_id": os.getpid(),
        "setup_fingerprint": setup_fingerprint,
        "remote": {
            "base_url": _safe_remote_url(remote_base_url),
            "text_model": remote_text_model.strip(),
            "image_model": remote_image_model.strip() or remote_text_model.strip(),
            "speech_model": remote_speech_model.strip() or remote_text_model.strip(),
        },
        "runtime": _runtime_details(torch),
        "packages": {
            name: _installed_version(name)
            for name in (
                "torch",
                "transformers",
                "flash-linear-attention",
                "causal-conv1d",
            )
        },
    }


def setup_colab_runtime(
    *,
    repo_dir: str | Path,
    used_commit: str,
    model_profile: str,
    save_to_google_drive: bool,
    remote_base_url: str = "",
    remote_text_model: str = "",
    remote_image_model: str = "",
    remote_speech_model: str = "",
    transformers_was_loaded: bool = False,
    engine_was_loaded: bool = False,
    engine_module_file: str = "",
    previous_commit: str = "",
) -> dict[str, Any]:
    """Install and verify the pulled M.M.M checkout in the current Colab process."""

    profile = model_profile.strip()
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported model profile {profile!r}; choose one of "
            f"{', '.join(sorted(SUPPORTED_PROFILES))}."
        )
    if profile == "remote_quality":
        remote_base_url = _validated_remote_url(remote_base_url)
        remote_text_model = remote_text_model.strip()
        if not remote_text_model:
            raise ValueError("remote_quality requires a text model name.")
    checkout = Path(repo_dir).resolve()
    commit = used_commit.strip()
    print("🔍 Validating checkout integrity...", flush=True)
    _validate_checkout(
        repo_dir=checkout,
        used_commit=commit,
        previous_commit=previous_commit,
        engine_was_loaded=engine_was_loaded,
        engine_module_file=engine_module_file,
    )
    os.chdir(checkout)

    torch = None
    if profile in LOCAL_PROFILES:
        print("🔧 Checking CUDA GPU availability...", flush=True)
        torch = _require_local_cuda()
    print("=" * 60, flush=True)
    _install_project(local_profile=profile in LOCAL_PROFILES)
    print("=" * 60, flush=True)
    if profile not in LOCAL_PROFILES:
        try:
            import torch as installed_torch
        except ImportError:
            installed_torch = None
        torch = installed_torch

    output_root = _configure_output(save_to_google_drive)
    os.environ["MMM_BLOCKBENCH_WORKSPACE_ROOT"] = "/content"
    os.environ["MMM_ECOSYSTEM_DISCOVERY"] = "auto"
    if profile == "remote_quality":
        _configure_remote(
            remote_base_url=remote_base_url,
            remote_text_model=remote_text_model,
            remote_image_model=remote_image_model,
            remote_speech_model=remote_speech_model,
        )

    fingerprint = setup_request_fingerprint(
        repo_dir=checkout,
        used_commit=commit,
        model_profile=profile,
        save_to_google_drive=save_to_google_drive,
        remote_base_url=remote_base_url,
        remote_text_model=remote_text_model,
        remote_image_model=remote_image_model,
        remote_speech_model=remote_speech_model,
    )
    receipt = _build_receipt(
        repo_dir=checkout,
        used_commit=commit,
        model_profile=profile,
        save_to_google_drive=save_to_google_drive,
        output_root=output_root,
        remote_base_url=remote_base_url,
        remote_text_model=remote_text_model,
        remote_image_model=remote_image_model,
        remote_speech_model=remote_speech_model,
        setup_fingerprint=fingerprint,
        torch=torch,
    )
    receipt_json = _canonical_json(receipt)
    os.environ["MMM_COLAB_SETUP_FINGERPRINT"] = fingerprint
    os.environ["MMM_COLAB_SETUP_RECEIPT"] = receipt_json

    runtime = receipt["runtime"]
    print("Setup source:", f"{SETUP_API_VERSION}@{commit[:12]}")
    print("Python:", runtime["python"])
    print("CUDA:", runtime["cuda_available"])
    if runtime["cuda_available"]:
        print("GPU:", runtime["gpu"])
        print(
            "VRAM free/total:",
            f"{runtime['vram_free_bytes'] / 2**30:.2f}/"
            f"{runtime['vram_total_bytes'] / 2**30:.2f} GiB",
        )
    print("Setup fingerprint:", fingerprint)
    return {
        "repo_dir": str(checkout),
        "used_commit": commit,
        "output_root": output_root,
        "setup_fingerprint": fingerprint,
        "receipt": receipt,
    }


def assert_setup_state(
    state: Mapping[str, Any],
    *,
    repo_dir: str | Path,
    used_commit: str,
    model_profile: str,
    save_to_google_drive: bool,
    remote_base_url: str = "",
    remote_text_model: str = "",
    remote_image_model: str = "",
    remote_speech_model: str = "",
) -> None:
    """Fail if config, checkout, process, or source changed after setup."""

    receipt = state.get("receipt")
    if not isinstance(receipt, Mapping):
        raise RuntimeError("Colab setup receipt is missing; rerun setup cell 2.")
    expected = setup_request_fingerprint(
        repo_dir=repo_dir,
        used_commit=used_commit,
        model_profile=model_profile,
        save_to_google_drive=save_to_google_drive,
        remote_base_url=remote_base_url,
        remote_text_model=remote_text_model,
        remote_image_model=remote_image_model,
        remote_speech_model=remote_speech_model,
    )
    actual_head = _git_head(Path(repo_dir).resolve())
    tracked_changes = _tracked_changes(Path(repo_dir).resolve())
    checks = {
        "state fingerprint": state.get("setup_fingerprint"),
        "receipt fingerprint": receipt.get("setup_fingerprint"),
        "environment fingerprint": os.environ.get("MMM_COLAB_SETUP_FINGERPRINT"),
    }
    mismatched = [name for name, value in checks.items() if value != expected]
    if mismatched:
        raise RuntimeError(
            "Colab configuration changed after setup ("
            + ", ".join(mismatched)
            + "). Rerun setup cell 2 before planning or building."
        )
    if model_profile.strip() == "remote_quality":
        _assert_remote_environment(
            remote_base_url=remote_base_url,
            remote_text_model=remote_text_model,
            remote_image_model=remote_image_model,
            remote_speech_model=remote_speech_model,
        )
    if actual_head != used_commit or receipt.get("used_commit") != used_commit:
        raise RuntimeError(
            "The Git checkout changed after setup. Rerun setup cell 2 before "
            "planning or building."
        )
    if tracked_changes:
        raise RuntimeError(
            "Tracked engine source changed after setup. Rerun setup cell 2 from "
            "a clean GitHub checkout."
        )
    _assert_loaded_engine_origin(Path(repo_dir).resolve())
    if receipt.get("process_id") != os.getpid():
        raise RuntimeError(
            "This setup receipt belongs to another Python runtime. Rerun setup "
            "cell 2."
        )
    script_path = Path(__file__).resolve()
    if receipt.get("setup_script_sha256") != _file_sha256(script_path):
        raise RuntimeError(
            "The source-owned Colab setup script changed after setup. Rerun "
            "setup cell 2."
        )
    if os.environ.get("MMM_COLAB_SETUP_RECEIPT") != _canonical_json(receipt):
        raise RuntimeError(
            "The Colab setup receipt was replaced or cleared. Rerun setup cell 2."
        )
