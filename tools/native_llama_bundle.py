from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable


BUNDLE_SCHEMA_VERSION = "mmm/native-llama-cuda-bundle-v2"
BUNDLE_RELEASE_TAG = "native-llama-b10375-cuda12.4-v2"
BUNDLE_RELEASE_BASE = (
    "https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode/releases/download/"
    + BUNDLE_RELEASE_TAG
)
BUNDLE_NAME_PREFIX = "llama-b10375-cuda12.4"
SUPPORTED_CUDA_ARCHES = frozenset({"75", "80", "89"})
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_name(cuda_arch: str) -> str:
    return f"{BUNDLE_NAME_PREFIX}-sm{cuda_arch}-linux-x86_64.tar.gz"


def _archive_url(cuda_arch: str) -> str:
    override = os.environ.get("MMM_LLAMA_PREBUILT_URL", "").strip()
    if override:
        return override
    return f"{BUNDLE_RELEASE_BASE}/{_asset_name(cuda_arch)}"


def _cache_root() -> Path:
    override = os.environ.get("MMM_LLAMA_PREBUILT_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (
        Path.home() / ".cache" / "mmm" / "llama-server" / BUNDLE_RELEASE_TAG
    ).resolve()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "M.M.M-Colab-native-llama/2"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as out:
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_ARCHIVE_BYTES:
                    raise RuntimeError(
                        "prebuilt archive exceeds the configured size limit"
                    )
            except ValueError:
                pass
        copied = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_ARCHIVE_BYTES:
                raise RuntimeError("prebuilt archive exceeds the configured size limit")
            out.write(chunk)


def _read_expected_sha256(url: str, destination: Path) -> str:
    _download(url + ".sha256", destination)
    token = destination.read_text(encoding="utf-8").strip().split()[0].lower()
    if len(token) != 64 or any(ch not in "0123456789abcdef" for ch in token):
        raise RuntimeError("prebuilt checksum file is malformed")
    return token


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"unsafe prebuilt archive member: {member.name}")
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe prebuilt archive path: {member.name}")
            resolved = (root / member_path).resolve()
            if not resolved.is_relative_to(root):
                raise RuntimeError(f"unsafe prebuilt archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise RuntimeError(
                    f"unsupported prebuilt archive member: {member.name}"
                )
        tar.extractall(root, members=members)


def _load_manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "prebuilt native llama manifest is missing or invalid"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("prebuilt native llama manifest must be an object")
    return payload


def _validate_bundle(root: Path, *, cuda_arch: str, source_ref: str) -> Path:
    root = root.resolve()
    manifest = _load_manifest(root)
    expected = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "llama_source_ref": source_ref,
        "cuda_arch": cuda_arch,
        "platform": "linux-x86_64",
    }
    for key, value in expected.items():
        if str(manifest.get(key, "")) != value:
            raise RuntimeError(
                f"prebuilt native llama manifest mismatch for {key}: "
                f"expected {value!r}, found {manifest.get(key)!r}"
            )
    if manifest.get("cuda_graphs") is not True:
        raise RuntimeError(
            "prebuilt native llama manifest requires cuda_graphs=true"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("prebuilt native llama manifest contains no files")
    for relative, expected_digest in files.items():
        if not isinstance(relative, str) or not isinstance(expected_digest, str):
            raise RuntimeError(
                "prebuilt native llama manifest file entry is invalid"
            )
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"prebuilt native llama file is missing: {relative}")
        if _sha256(path) != expected_digest.lower():
            raise RuntimeError(
                f"prebuilt native llama checksum mismatch: {relative}"
            )

    binary = root / "bin" / "llama-server"
    if not binary.is_file():
        raise RuntimeError("prebuilt native llama-server binary is missing")
    cuda_backends = [
        path
        for path in (root / "bin").glob("libggml-cuda.so*")
        if path.is_file()
    ]
    if not cuda_backends:
        raise RuntimeError("prebuilt native llama CUDA shared library is missing")
    binary.chmod(binary.stat().st_mode | 0o111)
    return binary.resolve()


def _prepend_library_path(directory: Path) -> None:
    value = str(directory.resolve())
    current = [
        part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part
    ]
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [value] + [part for part in current if part != value]
    )


def _verify_linkage(binary: Path) -> None:
    ldd = shutil.which("ldd")
    if not ldd:
        return
    completed = subprocess.run(
        [ldd, str(binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=20,
        env=os.environ.copy(),
    )
    if completed.returncode != 0 or "not found" in completed.stdout:
        tail = completed.stdout[-1600:].strip()
        raise RuntimeError(
            "prebuilt native llama shared-library linkage failed: " + tail
        )


def _cached_bundle(
    install_dir: Path,
    *,
    cuda_arch: str,
    source_ref: str,
    verify: Callable[[Path], tuple[bool, str]],
) -> Path | None:
    try:
        binary = _validate_bundle(
            install_dir,
            cuda_arch=cuda_arch,
            source_ref=source_ref,
        )
        _prepend_library_path(binary.parent)
        _verify_linkage(binary)
        ok, detail = verify(binary)
        if not ok:
            raise RuntimeError(detail)
        return binary
    except Exception:
        return None


def ensure_prebuilt_native_server(
    *,
    cuda_arch: str,
    source_ref: str,
    verify: Callable[[Path], tuple[bool, str]],
) -> str | None:
    """Install a verified native CUDA llama-server bundle, or return None for fallback."""

    if not _env_enabled("MMM_LLAMA_PREBUILT", True):
        return None
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        return None
    custom_url = bool(os.environ.get("MMM_LLAMA_PREBUILT_URL", "").strip())
    if cuda_arch not in SUPPORTED_CUDA_ARCHES and not custom_url:
        return None

    install_dir = _cache_root() / f"sm{cuda_arch}"
    cached = _cached_bundle(
        install_dir,
        cuda_arch=cuda_arch,
        source_ref=source_ref,
        verify=verify,
    )
    if cached is not None:
        os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] = "prebuilt-cache"
        return str(cached)

    cache_parent = install_dir.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    url = _archive_url(cuda_arch)
    with tempfile.TemporaryDirectory(
        prefix="native-llama-",
        dir=cache_parent,
    ) as temporary:
        temp = Path(temporary)
        archive = temp / _asset_name(cuda_arch)
        checksum = temp / (archive.name + ".sha256")
        expected_archive_sha = _read_expected_sha256(url, checksum)
        _download(url, archive)
        actual_archive_sha = _sha256(archive)
        if actual_archive_sha != expected_archive_sha:
            raise RuntimeError(
                "prebuilt native llama archive checksum mismatch: "
                f"expected {expected_archive_sha}, found {actual_archive_sha}"
            )
        extracted = temp / "extracted"
        _safe_extract(archive, extracted)
        binary = _validate_bundle(
            extracted,
            cuda_arch=cuda_arch,
            source_ref=source_ref,
        )
        _prepend_library_path(binary.parent)
        _verify_linkage(binary)
        ok, detail = verify(binary)
        if not ok:
            raise RuntimeError(
                "prebuilt native llama verification failed: " + detail
            )

        if install_dir.exists():
            shutil.rmtree(install_dir)
        os.replace(extracted, install_dir)

    final_binary = _validate_bundle(
        install_dir,
        cuda_arch=cuda_arch,
        source_ref=source_ref,
    )
    _prepend_library_path(final_binary.parent)
    _verify_linkage(final_binary)
    ok, detail = verify(final_binary)
    if not ok:
        raise RuntimeError(
            "installed prebuilt native llama verification failed: " + detail
        )
    os.environ["MMM_LLAMA_SERVER_DISTRIBUTION"] = "prebuilt-download"
    return str(final_binary)


__all__ = [
    "BUNDLE_RELEASE_TAG",
    "BUNDLE_SCHEMA_VERSION",
    "SUPPORTED_CUDA_ARCHES",
    "ensure_prebuilt_native_server",
]
