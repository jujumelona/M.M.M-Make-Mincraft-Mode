from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SETUP_API_VERSION = "mmm/colab-runtime-setup-v4-max-native"
RECEIPT_SCHEMA_VERSION = "mmm/colab-setup-receipt-v2"
REMOTE_PROFILE = "remote_quality"
REMOTE_TEXT_ROLES = ("PLANNER", "RESEARCH", "CODER", "CODER_SAFE", "VISION")
_REMOTE_PROFILE_ENV_NAMES = tuple(
    name
    for role in REMOTE_TEXT_ROLES
    for name in (
        f"MMM_{role}_BASE_URL",
        f"MMM_{role}_MODEL",
        f"MMM_{role}_API_KEY",
    )
) + (
    "MMM_IMAGE_BASE_URL",
    "MMM_IMAGE_MODEL",
    "MMM_IMAGE_API_KEY",
    "MMM_SPEECH_BASE_URL",
    "MMM_SPEECH_MODEL",
    "MMM_SPEECH_API_KEY",
)
_LOCAL_PROFILE_ENV_NAMES = (
    "LLAMA_SERVER_URL",
    "MMM_LLAMA_SERVER_BIN",
    "MMM_LLAMA_SERVER_DISTRIBUTION",
    "MMM_LLAMA_SERVER_SOURCE_DIR",
)
REMOTE_PROJECT_INSTALL_TARGET = ".[ui,rag,image,speech,production-audio]"
LOCAL_PROJECT_INSTALL_TARGET = ".[ui,local-model,rag,image,speech,production-audio]"

# Pinned official ggml-org/llama.cpp release commit. Local GGUF execution uses the
# native llama-server binary from the verified prebuilt bundle. Source compilation is
# emergency-only and must be explicitly enabled; there is no Python binding fallback.
LLAMA_SERVER_SOURCE_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_SERVER_SOURCE_REF = "1d2869c6e54d5003f3927a79efbca0fefa034a6d"
LLAMA_SERVER_DEFAULT_SOURCE_DIR = Path("/content/llama.cpp")
_NATIVE_VERIFY_CACHE: dict[tuple[object, ...], tuple[bool, str]] = {}
_NATIVE_VERIFY_CACHE_LIMIT = 16


def _is_local_profile(profile: object) -> bool:
    return str(profile or "").strip() != REMOTE_PROFILE


def _clear_inactive_profile_environment(*, local_profile: bool) -> None:
    """Remove state owned by the profile that is no longer active."""

    stale_names = _REMOTE_PROFILE_ENV_NAMES if local_profile else _LOCAL_PROFILE_ENV_NAMES
    for name in stale_names:
        os.environ.pop(name, None)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_remote_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.hostname:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
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
    local_profile = _is_local_profile(model_profile)
    request = {
        "setup_api_version": SETUP_API_VERSION,
        "repo_dir": str(Path(repo_dir).resolve()),
        "used_commit": used_commit.strip(),
        "model_profile": model_profile.strip(),
        "save_to_google_drive": bool(save_to_google_drive),
        "remote_base_url": "" if local_profile else remote_base_url.strip(),
        "remote_text_model": "" if local_profile else remote_text_model.strip(),
        "remote_image_model": "" if local_profile else remote_image_model.strip(),
        "remote_speech_model": "" if local_profile else remote_speech_model.strip(),
        "llama_server_source_ref": LLAMA_SERVER_SOURCE_REF if local_profile else "",
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


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _shutdown_loaded_managed_llama_server() -> bool:
    """Stop the old managed native server before purging a hot Colab engine.

    A source update can change server launch flags. Keeping the previous process alive
    while replacing its Python owner leaves LLAMA_SERVER_URL pointing at a server with
    stale capabilities and loses the process handle needed for a clean restart.
    """

    module = sys.modules.get("minecraft_mod_ai.llama_server_autotune")
    if module is None:
        return False
    shutdown = getattr(module, "_shutdown_managed_server", None)
    if not callable(shutdown):
        return False
    managed_url = str(getattr(module, "_MANAGED_URL", "") or "").strip()
    try:
        shutdown()
    except Exception as exc:
        raise RuntimeError(
            "Failed to stop the managed llama-server before engine reload. "
            "Restart the Colab runtime and rerun setup cell 2."
        ) from exc
    if managed_url and os.environ.get("LLAMA_SERVER_URL", "").strip() == managed_url:
        os.environ.pop("LLAMA_SERVER_URL", None)
    return True


def _reset_inactive_profile_state(*, local_profile: bool) -> None:
    """Quiesce runtime state owned by the profile being left."""

    if not local_profile:
        _shutdown_loaded_managed_llama_server()
    _clear_inactive_profile_environment(local_profile=local_profile)


def _validate_checkout(
    *,
    repo_dir: Path,
    used_commit: str,
    previous_commit: str,
    engine_was_loaded: bool,
    engine_module_file: str,
) -> None:
    del engine_module_file
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
    if engine_was_loaded:
        if not previous_commit or previous_commit.strip() != used_commit:
            print(
                f"engine reload: {previous_commit[:7] if previous_commit else 'old'} -> "
                f"{used_commit[:7]}",
                flush=True,
            )
        else:
            print(
                "engine reload: stopping the previous managed runtime before "
                "reapplying notebook settings",
                flush=True,
            )
        _shutdown_loaded_managed_llama_server()
        for name in list(sys.modules):
            if name == "minecraft_mod_ai" or name.startswith("minecraft_mod_ai."):
                sys.modules.pop(name, None)
        importlib.invalidate_caches()


def _require_local_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Select a Colab GPU runtime."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The selected local profile requires a Colab GPU runtime; CUDA is unavailable."
        )
    return torch


def _native_source_dir() -> Path:
    raw = os.environ.get("MMM_LLAMA_SERVER_SOURCE_DIR", "").strip()
    return (
        Path(raw).expanduser().resolve()
        if raw
        else LLAMA_SERVER_DEFAULT_SOURCE_DIR.resolve()
    )


def _native_server_candidates() -> list[Path]:
    values: list[Path] = []
    explicit = os.environ.get("MMM_LLAMA_SERVER_BIN", "").strip()
    if explicit:
        values.append(Path(explicit).expanduser())
    discovered = shutil.which("llama-server")
    if discovered:
        values.append(Path(discovered))
    source = _native_source_dir()
    values.append(source / "build" / "bin" / "llama-server")
    values.append(
        Path.home()
        / ".cache"
        / "mmm"
        / "llama.cpp"
        / "build"
        / "bin"
        / "llama-server"
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for value in values:
        try:
            key = str(value.expanduser().resolve(strict=False))
        except OSError:
            key = str(value.expanduser().absolute())
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _native_cuda_backend(binary: Path) -> Path | None:
    roots = [binary.parent, binary.parent.parent]
    for root in roots:
        for pattern in ("libggml-cuda.so", "libggml-cuda.so.*"):
            matches = sorted(path for path in root.rglob(pattern) if path.is_file())
            if matches:
                return matches[0]
    return None


def _native_verify_signature(binary: Path, backend: Path) -> tuple[object, ...]:
    binary = binary.resolve()
    backend = backend.resolve()
    binary_stat = binary.stat()
    backend_stat = backend.stat()
    return (
        str(binary),
        int(binary_stat.st_size),
        int(binary_stat.st_mtime_ns),
        str(backend),
        int(backend_stat.st_size),
        int(backend_stat.st_mtime_ns),
        os.environ.get("LD_LIBRARY_PATH", ""),
    )


def _verify_native_server(binary: Path) -> tuple[bool, str]:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return False, "binary missing or not executable"
    backend = _native_cuda_backend(binary)
    if backend is None:
        return False, "CUDA backend library is missing"
    try:
        signature = _native_verify_signature(binary, backend)
    except OSError as exc:
        return False, f"native server stat failed: {type(exc).__name__}: {exc}"
    cached = _NATIVE_VERIFY_CACHE.get(signature)
    if cached is not None:
        return cached
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=20,
        )
    except Exception as exc:
        return False, f"version probe failed: {type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        return False, "version probe failed: " + completed.stdout[-1000:]
    result = (True, f"binary={binary.name} cuda_backend={backend.name}")
    _NATIVE_VERIFY_CACHE[signature] = result
    while len(_NATIVE_VERIFY_CACHE) > _NATIVE_VERIFY_CACHE_LIMIT:
        _NATIVE_VERIFY_CACHE.pop(next(iter(_NATIVE_VERIFY_CACHE)))
    return result


def _find_verified_native_server() -> Path | None:
    for candidate in _native_server_candidates():
        ok, _detail = _verify_native_server(candidate)
        if ok:
            return candidate.resolve()
    return None


def _load_native_bundle_module() -> Any:
    name = "_mmm_native_llama_bundle"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().with_name("native_llama_bundle.py")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"native llama bundle loader is unavailable: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native llama bundle loader: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _ensure_prebuilt_native_server(*, cuda_arch: str) -> str | None:
    module = _load_native_bundle_module()
    ensure = getattr(module, "ensure_prebuilt_native_server", None)
    if not callable(ensure):
        raise RuntimeError("native llama bundle loader has no installer entry point")
    return ensure(
        cuda_arch=cuda_arch,
        source_ref=LLAMA_SERVER_SOURCE_REF,
        verify=_verify_native_server,
    )


def _run_logged(command: list[str], *, cwd: Path | None = None) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _prepare_native_source(source_dir: Path) -> None:
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (source_dir / ".git").is_dir():
        if source_dir.exists():
            shutil.rmtree(source_dir)
        print("native llama-server: fetching pinned source", flush=True)
        _run_logged(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                LLAMA_SERVER_SOURCE_REPOSITORY,
                str(source_dir),
            ]
        )
    current = ""
    try:
        current = _git_head(source_dir)
    except Exception:
        pass
    if current != LLAMA_SERVER_SOURCE_REF:
        print(
            "native llama-server: selecting pinned commit",
            LLAMA_SERVER_SOURCE_REF[:12],
            flush=True,
        )
        _run_logged(
            [
                "git",
                "-C",
                str(source_dir),
                "fetch",
                "--depth",
                "1",
                "origin",
                LLAMA_SERVER_SOURCE_REF,
            ]
        )
        _run_logged(
            [
                "git",
                "-C",
                str(source_dir),
                "checkout",
                "--detach",
                "FETCH_HEAD",
            ]
        )
    if _git_head(source_dir) != LLAMA_SERVER_SOURCE_REF:
        raise RuntimeError("native llama-server source commit verification failed")


def _ensure_native_server(torch: Any) -> str:
    existing = _find_verified_native_server()
    if existing is not None:
        resolved = str(existing)
        os.environ["MMM_LLAMA_SERVER_BIN"] = resolved
        os.environ.setdefault("MMM_LLAMA_SERVER_DISTRIBUTION", "existing")
        print("native llama-server: available", existing, flush=True)
        return resolved

    major, minor = torch.cuda.get_device_capability(0)
    cuda_arch = str(int(major) * 10 + int(minor))
    prebuilt_error: BaseException | None = None
    try:
        prebuilt = _ensure_prebuilt_native_server(cuda_arch=cuda_arch)
    except Exception as exc:
        prebuilt_error = exc
        prebuilt = None

    if prebuilt:
        resolved = str(Path(prebuilt).expanduser().resolve())
        ok, detail = _verify_native_server(Path(resolved))
        if not ok:
            raise RuntimeError(
                "prebuilt native llama-server failed final setup verification: " + detail
            )
        os.environ["MMM_LLAMA_SERVER_BIN"] = resolved
        print("native llama-server: using verified prebuilt", resolved, flush=True)
        return resolved

    if not _env_enabled("MMM_LLAMA_ALLOW_SOURCE_BUILD", False):
        cause = (
            f"{type(prebuilt_error).__name__}: {prebuilt_error}"
            if prebuilt_error is not None
            else "no compatible verified prebuilt bundle was returned"
        )
        raise RuntimeError(
            f"verified prebuilt native llama-server unavailable for CUDA SM{cuda_arch}; "
            "automatic source compilation is disabled to prevent long Colab rebuilds. "
            "Wait for/fix the matching prebuilt release asset. Emergency source compilation "
            "is available only with MMM_LLAMA_ALLOW_SOURCE_BUILD=1. Cause: "
            + cause
        ) from prebuilt_error

    print(
        "native llama-server: explicit emergency source-build fallback enabled",
        f"arch={cuda_arch}",
        flush=True,
    )
    for tool in ("git", "cmake", "nvcc"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"native llama-server CUDA build requires {tool!r}, but it is unavailable"
            )

    source_dir = _native_source_dir()
    _prepare_native_source(source_dir)
    build_dir = source_dir / "build"
    jobs = max(1, min(8, os.cpu_count() or 1))
    print(
        "native llama-server: configuring CUDA build",
        f"arch={cuda_arch}",
        f"jobs={jobs}",
        flush=True,
    )
    _run_logged(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            "-DGGML_CUDA_GRAPHS=ON",
            "-DGGML_CUDA_CUB_3DOT2=ON",
            "-DGGML_CUDA_FA=ON",
            "-DGGML_CUDA_FA_ALL_QUANTS=OFF",
            "-DGGML_LTO=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_SERVER=ON",
        ]
    )
    print("native llama-server: building", flush=True)
    _run_logged(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "llama-server",
            "-j",
            str(jobs),
        ]
    )
    binary = build_dir / "bin" / "llama-server"
    ok, detail = _verify_native_server(binary)
    if not ok:
        raise RuntimeError("native llama-server verification failed: " + detail)
    resolved = str(binary.resolve())
    os.environ["MMM_LLAMA_SERVER_BIN"] = resolved
    os.environ["MMM_LLAMA_SERVER_SOURCE_DIR"] = str(source_dir)
    os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] = "source-build"
    print("native llama-server: installed", detail, flush=True)
    return resolved


def _project_install_receipt_path() -> Path:
    return (Path.home() / ".cache" / "mmm" / "project-install-receipt.json").resolve()


def _project_install_fingerprint(target: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    payload = {
        "schema": "mmm/project-install-receipt-v1",
        "repo_root": str(repo_root),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "target": target,
        "pyproject_sha256": _file_sha256(pyproject),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _install_project(*, local_profile: bool) -> None:
    target = LOCAL_PROJECT_INSTALL_TARGET if local_profile else REMOTE_PROJECT_INSTALL_TARGET
    fingerprint = _project_install_fingerprint(target)
    receipt_path = _project_install_receipt_path()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception:
        receipt = {}
    if (
        receipt.get("schema_version") == "mmm/project-install-receipt-v1"
        and receipt.get("fingerprint") == fingerprint
        and _installed_version("mmm-make-mincraft-mode") is not None
    ):
        print("project dependencies: receipt hit; pip skipped", flush=True)
        return

    print("project dependencies: installing", target, flush=True)
    _run_logged(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "--no-build-isolation",
            "-e",
            target,
        ]
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    temporary.write_text(
        _canonical_json(
            {
                "schema_version": "mmm/project-install-receipt-v1",
                "fingerprint": fingerprint,
                "target": target,
            }
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt_path)
    print("project dependencies: installed", flush=True)


def _preflight_jdtls() -> str:
    from minecraft_mod_ai.jdtls_bootstrap import ensure_jdtls

    print("JDT LS: checking", flush=True)
    launcher = ensure_jdtls().resolve()
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RuntimeError(f"Managed JDT LS launcher is unavailable: {launcher}")
    resolved = str(launcher)
    os.environ["MMM_JDTLS_CMD"] = resolved
    print("JDT LS: available", resolved, flush=True)
    return resolved


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
    key_names = [
        *(f"MMM_{role}_API_KEY" for role in REMOTE_TEXT_ROLES),
        "MMM_IMAGE_API_KEY",
        "MMM_SPEECH_API_KEY",
    ]
    missing_keys = [name for name in key_names if not os.environ.get(name)]
    if mismatched or missing_keys:
        raise RuntimeError(
            "Remote model environment changed after setup. Rerun setup cell 2 before "
            "planning or building."
        )


def _runtime_details(torch: Any | None) -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "cpu_count": int(os.cpu_count() or 1),
    }
    try:
        memory: dict[str, int] = {}
        for raw_line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = raw_line.partition(":")
            if not separator:
                continue
            fields = raw_value.strip().split()
            if not fields:
                continue
            memory[key] = int(fields[0]) * 1024
        if memory.get("MemTotal", 0) > 0:
            details["system_ram_total_bytes"] = memory["MemTotal"]
            details["system_ram_available_bytes"] = memory.get(
                "MemAvailable",
                memory.get("MemFree", 0),
            )
    except (OSError, ValueError):
        pass
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
    llama_server_binary: str,
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    local_profile = _is_local_profile(model_profile)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "setup_api_version": SETUP_API_VERSION,
        "setup_script_sha256": _file_sha256(script_path),
        "repo_dir": str(repo_dir),
        "used_commit": used_commit,
        "model_profile": model_profile,
        "backend": "local_cuda" if _is_local_profile(model_profile) else "remote_api",
        "save_to_google_drive": bool(save_to_google_drive),
        "output_root": output_root,
        "process_id": os.getpid(),
        "setup_fingerprint": setup_fingerprint,
        "native_llama_server": {
            "source_ref": LLAMA_SERVER_SOURCE_REF if llama_server_binary else "",
            "binary": llama_server_binary,
        },
        "remote": {
            "base_url": "" if local_profile else _safe_remote_url(remote_base_url),
            "text_model": "" if local_profile else remote_text_model.strip(),
            "image_model": (
                ""
                if local_profile
                else remote_image_model.strip() or remote_text_model.strip()
            ),
            "speech_model": (
                ""
                if local_profile
                else remote_speech_model.strip() or remote_text_model.strip()
            ),
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

    del transformers_was_loaded
    profile = model_profile.strip()
    if not profile:
        raise ValueError("model_profile must be non-empty.")
    if profile == REMOTE_PROFILE:
        remote_base_url = _validated_remote_url(remote_base_url)
        remote_text_model = remote_text_model.strip()
        if not remote_text_model:
            raise ValueError("remote_quality requires a text model name.")

    checkout = Path(repo_dir).resolve()
    commit = used_commit.strip()
    print("checkout: validating", flush=True)
    _validate_checkout(
        repo_dir=checkout,
        used_commit=commit,
        previous_commit=previous_commit,
        engine_was_loaded=engine_was_loaded,
        engine_module_file=engine_module_file,
    )
    os.chdir(checkout)

    local_profile = _is_local_profile(profile)
    _reset_inactive_profile_state(local_profile=local_profile)
    torch = None
    llama_server_binary = ""
    if local_profile:
        print("CUDA: checking", flush=True)
        torch = _require_local_cuda()
        llama_server_binary = _ensure_native_server(torch)
    _install_project(local_profile=local_profile)
    _preflight_jdtls()
    if not local_profile:
        try:
            import torch as installed_torch
        except ImportError:
            installed_torch = None
        torch = installed_torch

    output_root = _configure_output(save_to_google_drive)
    os.environ["MMM_BLOCKBENCH_WORKSPACE_ROOT"] = "/content"
    os.environ["MMM_ECOSYSTEM_DISCOVERY"] = "auto"
    if profile == REMOTE_PROFILE:
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
        llama_server_binary=llama_server_binary,
    )
    receipt_json = _canonical_json(receipt)
    os.environ["MMM_COLAB_SETUP_FINGERPRINT"] = fingerprint
    os.environ["MMM_COLAB_SETUP_RECEIPT"] = receipt_json

    runtime = receipt["runtime"]
    print("Setup source:", f"{SETUP_API_VERSION}@{commit[:12]}")
    print("Python:", runtime["python"])
    print("CPU workers available:", runtime["cpu_count"])
    if runtime.get("system_ram_total_bytes"):
        print(
            "System RAM available/total:",
            f"{runtime.get('system_ram_available_bytes', 0) / 2**30:.2f}/"
            f"{runtime['system_ram_total_bytes'] / 2**30:.2f} GiB",
        )
    print("CUDA:", runtime["cuda_available"])
    if runtime["cuda_available"]:
        print("GPU:", runtime["gpu"])
        print(
            "VRAM free/total:",
            f"{runtime['vram_free_bytes'] / 2**30:.2f}/"
            f"{runtime['vram_total_bytes'] / 2**30:.2f} GiB",
        )
        print("llama-server:", llama_server_binary)
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
    """Fail if config, checkout, process, native server, or source changed after setup."""

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
    if model_profile.strip() == REMOTE_PROFILE:
        _assert_remote_environment(
            remote_base_url=remote_base_url,
            remote_text_model=remote_text_model,
            remote_image_model=remote_image_model,
            remote_speech_model=remote_speech_model,
        )
    else:
        native = receipt.get("native_llama_server")
        binary = str(native.get("binary", "")) if isinstance(native, Mapping) else ""
        ok, detail = _verify_native_server(Path(binary)) if binary else (False, "missing")
        if not ok or os.environ.get("MMM_LLAMA_SERVER_BIN") != binary:
            raise RuntimeError(
                "Native llama-server changed or is unavailable after setup: "
                + detail
                + ". Rerun setup cell 2."
            )
    if actual_head != used_commit or receipt.get("used_commit") != used_commit:
        raise RuntimeError(
            "The Git checkout changed after setup. Rerun setup cell 2 before planning "
            "or building."
        )
    if tracked_changes:
        raise RuntimeError(
            "Tracked engine source changed after setup. Rerun setup cell 2 from a "
            "clean GitHub checkout."
        )
    if receipt.get("process_id") != os.getpid():
        raise RuntimeError(
            "This setup receipt belongs to another Python runtime. Rerun setup cell 2."
        )
    script_path = Path(__file__).resolve()
    if receipt.get("setup_script_sha256") != _file_sha256(script_path):
        raise RuntimeError(
            "The source-owned Colab setup script changed after setup. Rerun setup cell 2."
        )
    if os.environ.get("MMM_COLAB_SETUP_RECEIPT") != _canonical_json(receipt):
        raise RuntimeError(
            "The Colab setup receipt was replaced or cleared. Rerun setup cell 2."
        )
