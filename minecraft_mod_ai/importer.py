"""Safe, inspection-only import of an existing Minecraft mod ZIP.

The importer deliberately does not run Gradle wrappers, scripts, JARs, or any
other archive content.  It validates the complete ZIP inventory before
optionally extracting regular files into a newly-created directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import unicodedata
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .spec import Proposal, ProposalStatus, canonical_json


def _host_limit(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a non-negative integer.") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be a non-negative integer.")
    return value


# Zero means no project-wide product cap. Operators may set finite host quotas.
MAX_ARCHIVE_ENTRIES = _host_limit("MMM_IMPORT_MAX_ENTRIES", 0)
MAX_TOTAL_UNCOMPRESSED_BYTES = _host_limit("MMM_IMPORT_MAX_TOTAL_BYTES", 0)
MAX_SINGLE_FILE_BYTES = _host_limit(
    "MMM_IMPORT_MAX_SINGLE_FILE_BYTES",
    512 * 1024 * 1024,
)
MAX_JSON_BYTES = _host_limit("MMM_IMPORT_MAX_JSON_BYTES", 8 * 1024 * 1024)
MAX_NESTED_JAR_ENTRIES = _host_limit("MMM_IMPORT_MAX_JAR_ENTRIES", 0)
MAX_NESTED_JAR_UNCOMPRESSED_BYTES = _host_limit(
    "MMM_IMPORT_MAX_JAR_BYTES",
    0,
)
MAX_NESTED_SOURCE_ARCHIVES = _host_limit(
    "MMM_IMPORT_MAX_NESTED_SOURCE_ARCHIVES",
    0,
)

_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_EXTRACT_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_DEVICES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_FORBIDDEN_CACHE_PARTS = frozenset({".git", ".gradle", "build", "run"})
_CREDENTIAL_DIRS = frozenset(
    {".aws", ".azure", ".gnupg", ".kube", ".ssh", "credentials", "secrets"}
)
_CREDENTIAL_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "service-account.json",
        "service_account.json",
    }
)
_CREDENTIAL_SUFFIXES = (
    ".jks",
    ".key",
    ".keystore",
    ".p12",
    ".pem",
    ".pfx",
)
_SOURCE_SUFFIXES = frozenset({".java", ".kt", ".kts", ".scala"})
_WORLD_SUFFIXES = frozenset(
    {".dat", ".mca", ".mcstructure", ".nbt", ".schem", ".schematic"}
)


class ExistingProjectImportError(ValueError):
    """Raised when an existing-project ZIP fails the fail-closed policy."""


_CapturedValue = bytes | Path


@dataclass(frozen=True)
class ExistingProjectReport:
    """Serializable inventory of an inspected, but never executed, archive."""

    archive_name: str
    archive_sha256: str
    source_snapshot_hash: str
    file_count: int
    total_uncompressed_bytes: int
    root_name: str | None
    input_kind: str
    has_sources: bool
    has_gradle_project: bool
    jar_only: bool
    loader: str | None
    mod_id: str | None
    mod_name: str | None
    mod_version: str | None
    minecraft_version: str | None
    minecraft_versions: tuple[str, ...]
    fabric_metadata_paths: tuple[str, ...]
    gradle_files: tuple[str, ...]
    source_files: tuple[str, ...]
    mixin_files: tuple[str, ...]
    access_widener_files: tuple[str, ...]
    asset_files: tuple[str, ...]
    world_files: tuple[str, ...]
    jar_files: tuple[str, ...]
    release_bundles: tuple[str, ...]
    embedded_proposal_path: str | None
    embedded_proposal_status: str
    embedded_approval_hash: str | None
    trusted_generated_source: bool
    extracted_to: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return _json_compatible(asdict(self))


@dataclass(frozen=True)
class _ArchiveFile:
    info: zipfile.ZipInfo
    archive_path: str
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _FabricMetadata:
    path: str
    mod_id: str | None
    name: str | None
    version: str | None
    minecraft_versions: tuple[str, ...]
    mixins: tuple[str, ...]
    access_widener: str | None


@dataclass(frozen=True)
class _NestedSourceInventory:
    metadata: tuple[_FabricMetadata, ...] = ()
    metadata_paths: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    gradle_files: tuple[str, ...] = ()
    mixin_files: tuple[str, ...] = ()
    access_widener_files: tuple[str, ...] = ()
    asset_files: tuple[str, ...] = ()
    world_files: tuple[str, ...] = ()
    jar_files: tuple[str, ...] = ()


def inspect_existing_project_archive(
    archive_path: str | Path,
    *,
    extract_root: str | Path | None = None,
    expected_archive_sha256: str | None = None,
) -> ExistingProjectReport:
    """Inspect an existing mod/project ZIP without executing its contents.

    If ``extract_root`` is supplied, regular files are copied into an
    archive-specific child directory using a staged atomic rename. A completed
    extraction may be reused after its receipt and original file hashes are
    revalidated. Incomplete deterministic destinations are preserved under a
    distinct ``.incomplete-N`` name before retrying.

    When ``expected_archive_sha256`` is supplied, it is compared with the hash
    of the already-open archive stream before ZIP parsing or extraction. This
    binds the bytes being inspected and extracted to the approved input.
    """

    path = Path(archive_path)
    if expected_archive_sha256 is not None and not _SHA256.fullmatch(
        expected_archive_sha256
    ):
        raise ExistingProjectImportError(
            "expected_archive_sha256 must be a lowercase sha256: digest."
        )
    if path.suffix.casefold() != ".zip":
        raise ExistingProjectImportError("Existing projects must be supplied as a .zip archive.")
    if not path.is_file() or path.is_symlink():
        raise ExistingProjectImportError(f"Archive does not exist or is not a file: {path}")

    capture_spool = tempfile.TemporaryDirectory(prefix="mmm-import-capture-")
    capture_root = Path(capture_spool.name).resolve()
    with path.open("rb") as archive_stream:
        archive_sha256 = _hash_stream(archive_stream)
        qualified_archive_sha256 = f"sha256:{archive_sha256}"
        if (
            expected_archive_sha256 is not None
            and qualified_archive_sha256 != expected_archive_sha256
        ):
            raise ExistingProjectImportError(
                "Existing input bytes changed after complete-plan approval."
            )
        archive_stream.seek(0)
        try:
            with zipfile.ZipFile(archive_stream, "r") as archive:
                (
                    files,
                    captured,
                    root_name,
                    total_uncompressed,
                ) = _inspect_zip(
                    archive,
                    spool_root=capture_root,
                )

                warnings: list[str] = [
                    "User-supplied archives are untrusted input and never become "
                    "authoritative RAG or approval evidence."
                ]
                metadata, metadata_paths, nested_inventory = _inspect_fabric_metadata(
                    files, captured, warnings
                )
                nested_source = _inspect_nested_source_archives(
                    files,
                    captured,
                    warnings,
                    spool_root=capture_root,
                )
                metadata = (*metadata, *nested_source.metadata)
                metadata_paths = _sorted_paths(
                    metadata_paths,
                    nested_source.metadata_paths,
                )
                proposal_path, proposal_status, approval_hash = _inspect_embedded_proposal(
                    files, captured, warnings
                )

                relative_paths = tuple(file.relative_path for file in files)
                nested_mixins, nested_wideners, nested_assets = nested_inventory
                source_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_source_path(item)
                    ),
                    nested_source.source_files,
                )
                gradle_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_gradle_path(item)
                    ),
                    nested_source.gradle_files,
                )
                jar_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_mod_jar_path(item)
                    ),
                    nested_source.jar_files,
                )
                mixin_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_mixin_path(item)
                    ),
                    nested_mixins,
                    nested_source.mixin_files,
                    *(entry.mixins for entry in metadata),
                )
                access_widener_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_access_widener_path(item)
                    ),
                    nested_wideners,
                    nested_source.access_widener_files,
                    *(
                        (entry.access_widener,)
                        for entry in metadata
                        if entry.access_widener
                    ),
                )
                asset_files = _sorted_paths(
                    (
                        item
                        for item in relative_paths
                        if _is_asset_path(item)
                    ),
                    nested_assets,
                    nested_source.asset_files,
                )
                world_files = _sorted_paths(
                    (item for item in relative_paths if _is_world_path(item)),
                    nested_source.world_files,
                )
                release_bundles = _sorted_paths(
                    item for item in relative_paths if _is_release_bundle(item)
                )

                has_sources = bool(source_files)
                has_gradle_project = bool(gradle_files)
                jar_only = bool(jar_files) and not has_sources and not has_gradle_project
                if has_sources and jar_files:
                    input_kind = "source_and_release"
                elif has_sources or has_gradle_project:
                    input_kind = "source_project"
                elif jar_only:
                    input_kind = "jar_only"
                else:
                    input_kind = "unknown"

                primary_metadata = _select_primary_metadata(metadata)
                snapshot_hash = _source_snapshot_hash(files)
                extracted_to = (
                    _extract_validated_archive(
                        archive,
                        files,
                        extract_root=Path(extract_root),
                        archive_stem=path.stem,
                        snapshot_hash=snapshot_hash,
                        archive_sha256=qualified_archive_sha256,
                    )
                    if extract_root is not None
                    else None
                )
        except zipfile.BadZipFile as exc:
            raise ExistingProjectImportError("The supplied file is not a valid ZIP archive.") from exc
        except (NotImplementedError, RuntimeError) as exc:
            raise ExistingProjectImportError(
                "The ZIP uses encryption or an unsupported compression method."
            ) from exc

    capture_spool.cleanup()
    return ExistingProjectReport(
        archive_name=path.name,
        archive_sha256=qualified_archive_sha256,
        source_snapshot_hash=snapshot_hash,
        file_count=len(files),
        total_uncompressed_bytes=total_uncompressed,
        root_name=root_name,
        input_kind=input_kind,
        has_sources=has_sources,
        has_gradle_project=has_gradle_project,
        jar_only=jar_only,
        loader="fabric" if metadata_paths else None,
        mod_id=primary_metadata.mod_id if primary_metadata else None,
        mod_name=primary_metadata.name if primary_metadata else None,
        mod_version=primary_metadata.version if primary_metadata else None,
        minecraft_version=(
            primary_metadata.minecraft_versions[0]
            if primary_metadata and primary_metadata.minecraft_versions
            else None
        ),
        minecraft_versions=(
            primary_metadata.minecraft_versions if primary_metadata else ()
        ),
        fabric_metadata_paths=metadata_paths,
        gradle_files=gradle_files,
        source_files=source_files,
        mixin_files=mixin_files,
        access_widener_files=access_widener_files,
        asset_files=asset_files,
        world_files=world_files,
        jar_files=jar_files,
        release_bundles=release_bundles,
        embedded_proposal_path=proposal_path,
        embedded_proposal_status=proposal_status,
        embedded_approval_hash=approval_hash,
        trusted_generated_source=False,
        extracted_to=extracted_to,
        warnings=tuple(warnings),
    )


def _inspect_zip(
    archive: zipfile.ZipFile,
    *,
    spool_root: Path,
) -> tuple[list[_ArchiveFile], dict[str, _CapturedValue], str | None, int]:
    infos = archive.infolist()
    if MAX_ARCHIVE_ENTRIES and len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ExistingProjectImportError(
            f"Archive has too many entries ({len(infos)} > {MAX_ARCHIVE_ENTRIES})."
        )

    normalized: list[tuple[zipfile.ZipInfo, str, bool]] = []
    seen: dict[str, str] = {}
    file_paths: list[str] = []
    total_uncompressed = 0
    for info in infos:
        normalized_path, is_directory = _validate_member(info)
        collision_key = unicodedata.normalize("NFKC", normalized_path).casefold()
        if collision_key in seen:
            raise ExistingProjectImportError(
                "Archive contains duplicate or case-colliding paths: "
                f"{seen[collision_key]!r} and {normalized_path!r}."
            )
        seen[collision_key] = normalized_path
        normalized.append((info, normalized_path, is_directory))
        if is_directory:
            continue
        if MAX_SINGLE_FILE_BYTES and info.file_size > MAX_SINGLE_FILE_BYTES:
            raise ExistingProjectImportError(
                f"Archive entry exceeds the single-file limit: {normalized_path}"
            )
        total_uncompressed += info.file_size
        if (
            MAX_TOTAL_UNCOMPRESSED_BYTES
            and total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES
        ):
            raise ExistingProjectImportError(
                "Archive exceeds the total uncompressed-size limit."
            )
        file_paths.append(normalized_path)

    if not file_paths:
        raise ExistingProjectImportError("Archive contains no regular files.")
    _reject_file_directory_conflicts(normalized)
    root_name = _common_wrapper_root(file_paths)

    files: list[_ArchiveFile] = []
    captured: dict[str, _CapturedValue] = {}
    for info, archive_member, is_directory in normalized:
        if is_directory:
            continue
        relative = _strip_wrapper_root(archive_member, root_name)
        capture = _should_capture(relative)
        digest, data = _read_and_hash_member(
            archive,
            info,
            capture=capture,
            spool_root=spool_root,
        )
        files.append(
            _ArchiveFile(
                info=info,
                archive_path=archive_member,
                relative_path=relative,
                size=info.file_size,
                sha256=digest,
            )
        )
        if data is not None:
            captured[relative] = data
    files.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
    return files, captured, root_name, total_uncompressed


def _validate_member(info: zipfile.ZipInfo) -> tuple[str, bool]:
    raw = info.filename
    if not raw or "\x00" in raw:
        raise ExistingProjectImportError("Archive contains an empty or NUL-containing path.")
    portable = unicodedata.normalize("NFC", raw.replace("\\", "/"))
    if portable.startswith(("/", "//")) or _DRIVE_PATH.match(portable):
        raise ExistingProjectImportError(f"Absolute archive path is forbidden: {raw!r}")

    is_directory = info.is_dir() or portable.endswith("/")
    portable = portable.rstrip("/")
    parts = PurePosixPath(portable).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ExistingProjectImportError(f"Unsafe archive path: {raw!r}")
    for part in parts:
        if ":" in part or part.endswith((" ", ".")):
            raise ExistingProjectImportError(f"Non-portable archive path: {raw!r}")
        device_name = part.split(".", 1)[0].casefold()
        if device_name in _WINDOWS_DEVICES:
            raise ExistingProjectImportError(f"Windows device path is forbidden: {raw!r}")

    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        raise ExistingProjectImportError(f"Symbolic links are forbidden: {raw!r}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ExistingProjectImportError(f"Non-regular ZIP entry is forbidden: {raw!r}")
    if info.flag_bits & 0x1:
        raise ExistingProjectImportError(f"Encrypted ZIP entries are forbidden: {raw!r}")

    lowered_parts = tuple(part.casefold() for part in parts)
    if any(part in _FORBIDDEN_CACHE_PARTS for part in lowered_parts):
        raise ExistingProjectImportError(
            f"VCS/cache/build/run content is not accepted: {portable!r}"
        )
    if _is_credential_path(lowered_parts):
        raise ExistingProjectImportError(
            f"Credential or private-key-like path is not accepted: {portable!r}"
        )
    return "/".join(parts), is_directory


def _is_credential_path(lowered_parts: tuple[str, ...]) -> bool:
    if any(part in _CREDENTIAL_DIRS for part in lowered_parts):
        return True
    name = lowered_parts[-1]
    if name in _CREDENTIAL_NAMES or name.startswith(".env."):
        return True
    if name.endswith(_CREDENTIAL_SUFFIXES):
        return True
    stem = name.rsplit(".", 1)[0]
    return (
        stem in {"credential", "credentials", "private_key", "private-key", "secret", "secrets"}
        or stem.startswith(("credentials-", "private-key-", "private_key-", "secret-"))
    )


def _reject_file_directory_conflicts(
    entries: list[tuple[zipfile.ZipInfo, str, bool]],
) -> None:
    file_keys = {
        path.casefold()
        for _, path, is_directory in entries
        if not is_directory
    }
    for file_path in file_keys:
        parts = file_path.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in file_keys:
                raise ExistingProjectImportError(
                    f"A regular file is also used as a directory: {file_path!r}"
                )


def _common_wrapper_root(file_paths: list[str]) -> str | None:
    split = [path.split("/") for path in file_paths]
    first = split[0][0]
    if all(len(parts) > 1 and parts[0] == first for parts in split):
        return first
    return None


def _strip_wrapper_root(path: str, root_name: str | None) -> str:
    if root_name is None:
        return path
    prefix = f"{root_name}/"
    if not path.startswith(prefix):
        raise ExistingProjectImportError("Archive wrapper-root detection became inconsistent.")
    return path[len(prefix):]


def _should_capture(relative_path: str) -> bool:
    lowered = relative_path.casefold()
    suffix = PurePosixPath(lowered).suffix
    return (
        lowered == ".minecraft_ai/proposal.approved.json"
        or lowered.endswith("/.minecraft_ai/proposal.approved.json")
        or PurePosixPath(lowered).name == "fabric.mod.json"
        or suffix in {".jar", ".zip"}
    )


def _read_and_hash_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    capture: bool,
    spool_root: Path,
) -> tuple[str, _CapturedValue | None]:
    digest = hashlib.sha256()
    spool_archive = (
        capture
        and PurePosixPath(info.filename.casefold()).suffix in {".jar", ".zip"}
    )
    data = bytearray() if capture and not spool_archive else None
    spool_path = (
        spool_root / f"{uuid.uuid4().hex}.archive"
        if spool_archive
        else None
    )
    spool_stream = spool_path.open("xb") if spool_path is not None else None
    actual_size = 0
    try:
        with archive.open(info, "r") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                actual_size += len(chunk)
                if (
                    (
                        MAX_SINGLE_FILE_BYTES
                        and actual_size > MAX_SINGLE_FILE_BYTES
                    )
                    or actual_size > info.file_size
                ):
                    raise ExistingProjectImportError(
                        "Archive entry expanded beyond its declared or "
                        f"allowed size: {info.filename!r}"
                    )
                digest.update(chunk)
                if data is not None:
                    data.extend(chunk)
                if spool_stream is not None:
                    spool_stream.write(chunk)
        if spool_stream is not None:
            spool_stream.flush()
            os.fsync(spool_stream.fileno())
    except BaseException:
        if spool_stream is not None:
            spool_stream.close()
        if spool_path is not None:
            spool_path.unlink(missing_ok=True)
        raise
    finally:
        if spool_stream is not None and not spool_stream.closed:
            spool_stream.close()
    if actual_size != info.file_size:
        raise ExistingProjectImportError(
            f"Archive entry size mismatch: {info.filename!r}"
        )
    if spool_path is not None:
        return digest.hexdigest(), spool_path
    return digest.hexdigest(), bytes(data) if data is not None else None


def _source_snapshot_hash(files: list[_ArchiveFile]) -> str:
    manifest = [
        {"path": item.relative_path, "sha256": item.sha256}
        for item in sorted(files, key=lambda file: file.relative_path.encode("utf-8"))
    ]
    payload = {
        "schema": "minecraft-mod-ai/source-snapshot-v1",
        "files": manifest,
    }
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"


def _captured_bytes(
    value: _CapturedValue,
    *,
    context: str,
) -> bytes:
    if isinstance(value, bytes):
        return value
    if (
        not value.is_file()
        or value.is_symlink()
        or (MAX_JSON_BYTES and value.stat().st_size > MAX_JSON_BYTES)
    ):
        raise ExistingProjectImportError(
            f"Captured metadata is missing, unsafe, or too large: {context}"
        )
    return value.read_bytes()


def _zip_source(value: _CapturedValue) -> Path | io.BytesIO:
    if isinstance(value, Path):
        if not value.is_file() or value.is_symlink():
            raise ExistingProjectImportError(
                "Captured nested archive is missing or unsafe."
            )
        return value
    return io.BytesIO(value)


def _inspect_fabric_metadata(
    files: list[_ArchiveFile],
    captured: dict[str, _CapturedValue],
    warnings: list[str],
) -> tuple[
    tuple[_FabricMetadata, ...],
    tuple[str, ...],
    tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
]:
    metadata: list[_FabricMetadata] = []
    metadata_paths: list[str] = []
    nested_mixins: list[str] = []
    nested_wideners: list[str] = []
    nested_assets: list[str] = []

    for file in files:
        lowered = file.relative_path.casefold()
        if PurePosixPath(lowered).name == "fabric.mod.json":
            metadata_paths.append(file.relative_path)
            parsed = _parse_fabric_json(
                _captured_bytes(
                    captured[file.relative_path],
                    context=file.relative_path,
                ),
                file.relative_path,
                warnings,
            )
            if parsed is not None:
                metadata.append(parsed)
        if _is_mod_jar_path(lowered):
            jar_data = captured[file.relative_path]
            jar_result = _inspect_nested_jar(file.relative_path, jar_data, warnings)
            if jar_result is None:
                continue
            jar_metadata, jar_mixins, jar_wideners, jar_assets = jar_result
            if jar_metadata is not None:
                metadata.append(jar_metadata)
                metadata_paths.append(jar_metadata.path)
            nested_mixins.extend(jar_mixins)
            nested_wideners.extend(jar_wideners)
            nested_assets.extend(jar_assets)

    if len(metadata_paths) > 1:
        warnings.append(
            "Multiple fabric.mod.json files were found; the source-tree metadata is "
            "preferred over release-JAR metadata."
        )
    return (
        tuple(metadata),
        _sorted_paths(metadata_paths),
        (
            _sorted_paths(nested_mixins),
            _sorted_paths(nested_wideners),
            _sorted_paths(nested_assets),
        ),
    )


def _inspect_nested_source_archives(
    files: list[_ArchiveFile],
    captured: dict[str, _CapturedValue],
    warnings: list[str],
    *,
    spool_root: Path,
) -> _NestedSourceInventory:
    """Inventory source ZIPs embedded in a generated release bundle.

    A release produced by this project stores editable source as
    ``source/<release>-source.zip``. Treating the outer archive as JAR-only
    would lose the exact revision input described by the architecture PDF.
    Nested source is therefore validated with the same path, credential, size,
    and symlink policy, but it is never extracted or executed.
    """

    candidates = [
        file
        for file in files
        if _is_nested_source_archive_path(file.relative_path)
    ]
    if (
        MAX_NESTED_SOURCE_ARCHIVES
        and len(candidates) > MAX_NESTED_SOURCE_ARCHIVES
    ):
        raise ExistingProjectImportError(
            "Release bundle contains too many nested source archives."
        )
    if not candidates:
        return _NestedSourceInventory()

    metadata: list[_FabricMetadata] = []
    metadata_paths: list[str] = []
    source_files: list[str] = []
    gradle_files: list[str] = []
    mixin_files: list[str] = []
    access_widener_files: list[str] = []
    asset_files: list[str] = []
    world_files: list[str] = []
    jar_files: list[str] = []
    combined_uncompressed = 0

    for candidate in candidates:
        data = captured.get(candidate.relative_path)
        if data is None:
            raise ExistingProjectImportError(
                f"Nested source archive was not captured: {candidate.relative_path}"
            )
        try:
            with zipfile.ZipFile(_zip_source(data), "r") as nested:
                (
                    nested_files,
                    nested_captured,
                    _,
                    nested_total,
                ) = _inspect_zip(
                    nested,
                    spool_root=spool_root,
                )
        except zipfile.BadZipFile as exc:
            raise ExistingProjectImportError(
                f"Nested source archive is not a valid ZIP: {candidate.relative_path}"
            ) from exc
        except (NotImplementedError, RuntimeError) as exc:
            raise ExistingProjectImportError(
                "Nested source archive is encrypted or uses unsupported compression: "
                f"{candidate.relative_path}"
            ) from exc

        combined_uncompressed += nested_total
        if (
            MAX_TOTAL_UNCOMPRESSED_BYTES
            and combined_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES
        ):
            raise ExistingProjectImportError(
                "Nested source archives exceed the combined uncompressed-size limit."
            )

        prefix = f"{candidate.relative_path}!/"

        def nested_path(value: str) -> str:
            return f"{prefix}{value}"

        nested_metadata, nested_metadata_paths, nested_special = (
            _inspect_fabric_metadata(nested_files, nested_captured, warnings)
        )
        nested_mixins, nested_wideners, nested_assets = nested_special
        for entry in nested_metadata:
            metadata.append(
                _FabricMetadata(
                    path=nested_path(entry.path),
                    mod_id=entry.mod_id,
                    name=entry.name,
                    version=entry.version,
                    minecraft_versions=entry.minecraft_versions,
                    mixins=entry.mixins,
                    access_widener=entry.access_widener,
                )
            )
        metadata_paths.extend(nested_path(value) for value in nested_metadata_paths)

        relative_paths = tuple(file.relative_path for file in nested_files)
        source_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_source_path(value)
        )
        gradle_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_gradle_path(value)
        )
        mixin_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_mixin_path(value)
        )
        mixin_files.extend(nested_path(value) for value in nested_mixins)
        access_widener_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_access_widener_path(value)
        )
        access_widener_files.extend(
            nested_path(value) for value in nested_wideners
        )
        asset_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_asset_path(value)
        )
        asset_files.extend(nested_path(value) for value in nested_assets)
        world_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_world_path(value)
        )
        jar_files.extend(
            nested_path(value)
            for value in relative_paths
            if _is_mod_jar_path(value)
        )

    warnings.append(
        "Nested release-bundle source ZIPs were inventoried as untrusted data "
        "without extraction or execution."
    )
    return _NestedSourceInventory(
        metadata=tuple(metadata),
        metadata_paths=_sorted_paths(metadata_paths),
        source_files=_sorted_paths(source_files),
        gradle_files=_sorted_paths(gradle_files),
        mixin_files=_sorted_paths(mixin_files),
        access_widener_files=_sorted_paths(access_widener_files),
        asset_files=_sorted_paths(asset_files),
        world_files=_sorted_paths(world_files),
        jar_files=_sorted_paths(jar_files),
    )


def _is_nested_source_archive_path(path: str) -> bool:
    portable = PurePosixPath(path.casefold())
    if portable.suffix != ".zip":
        return False
    return (
        portable.name.endswith("-source.zip")
        or "source" in portable.parts[:-1]
        or "sources" in portable.parts[:-1]
    )


def _inspect_nested_jar(
    jar_path: str,
    data: _CapturedValue,
    warnings: list[str],
) -> tuple[_FabricMetadata | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    try:
        with zipfile.ZipFile(_zip_source(data), "r") as jar:
            infos = jar.infolist()
            if MAX_NESTED_JAR_ENTRIES and len(infos) > MAX_NESTED_JAR_ENTRIES:
                raise ExistingProjectImportError(
                    f"Nested JAR has too many entries: {jar_path}"
                )
            declared_total = sum(info.file_size for info in infos if not info.is_dir())
            if (
                MAX_NESTED_JAR_UNCOMPRESSED_BYTES
                and declared_total > MAX_NESTED_JAR_UNCOMPRESSED_BYTES
            ):
                raise ExistingProjectImportError(
                    f"Nested JAR exceeds the uncompressed-size inventory limit: {jar_path}"
                )
            names: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                normalized, is_directory = _validate_nested_jar_member(info, jar_path)
                if is_directory:
                    continue
                key = normalized.casefold()
                if key in names:
                    raise ExistingProjectImportError(
                        f"Nested JAR contains duplicate paths: {jar_path}!/{normalized}"
                    )
                names[key] = info

            mixins = _sorted_paths(
                f"{jar_path}!/{name}"
                for name in names
                if _is_mixin_path(name)
            )
            wideners = _sorted_paths(
                f"{jar_path}!/{name}"
                for name in names
                if _is_access_widener_path(name)
            )
            assets = _sorted_paths(
                f"{jar_path}!/{name}"
                for name in names
                if _is_asset_path(name)
            )
            metadata_info = names.get("fabric.mod.json")
            if metadata_info is None:
                return None, mixins, wideners, assets
            if metadata_info.file_size > MAX_JSON_BYTES:
                raise ExistingProjectImportError(
                    f"Nested fabric.mod.json is too large: {jar_path}"
                )
            metadata_bytes = jar.read(metadata_info)
            parsed = _parse_fabric_json(
                metadata_bytes, f"{jar_path}!/fabric.mod.json", warnings
            )
            return parsed, mixins, wideners, assets
    except zipfile.BadZipFile:
        warnings.append(f"JAR inventory could not be read as ZIP: {jar_path}")
        return None
    except (NotImplementedError, RuntimeError) as exc:
        raise ExistingProjectImportError(
            f"Nested JAR is encrypted or uses unsupported compression: {jar_path}"
        ) from exc


def _validate_nested_jar_member(
    info: zipfile.ZipInfo,
    jar_path: str,
) -> tuple[str, bool]:
    raw = info.filename
    portable = unicodedata.normalize("NFC", raw.replace("\\", "/"))
    if (
        not portable
        or "\x00" in portable
        or portable.startswith(("/", "//"))
        or _DRIVE_PATH.match(portable)
    ):
        raise ExistingProjectImportError(f"Unsafe path inside JAR: {jar_path}!/{raw}")
    is_directory = info.is_dir() or portable.endswith("/")
    portable = portable.rstrip("/")
    parts = PurePosixPath(portable).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ExistingProjectImportError(f"Unsafe path inside JAR: {jar_path}!/{raw}")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_IFMT(mode) == stat.S_IFLNK:
        raise ExistingProjectImportError(f"Symlink inside JAR is forbidden: {jar_path}!/{raw}")
    if info.flag_bits & 0x1:
        raise ExistingProjectImportError(f"Encrypted JAR entry is forbidden: {jar_path}!/{raw}")
    return "/".join(parts), is_directory


def _parse_fabric_json(
    data: bytes,
    metadata_path: str,
    warnings: list[str],
) -> _FabricMetadata | None:
    if len(data) > MAX_JSON_BYTES:
        warnings.append(f"fabric.mod.json exceeds the metadata parse limit: {metadata_path}")
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top level is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"Invalid fabric.mod.json at {metadata_path}: {exc}")
        return None

    mixins: list[str] = []
    raw_mixins = payload.get("mixins", [])
    if isinstance(raw_mixins, list):
        for item in raw_mixins:
            if isinstance(item, str):
                mixins.append(item)
            elif isinstance(item, dict) and isinstance(item.get("config"), str):
                mixins.append(item["config"])

    depends = payload.get("depends", {})
    minecraft = depends.get("minecraft") if isinstance(depends, dict) else None
    minecraft_versions = _string_values(minecraft)
    return _FabricMetadata(
        path=metadata_path,
        mod_id=_plain_string(payload.get("id")),
        name=_plain_string(payload.get("name")),
        version=_plain_string(payload.get("version")),
        minecraft_versions=minecraft_versions,
        mixins=tuple(mixins),
        access_widener=_plain_string(payload.get("accessWidener")),
    )


def _select_primary_metadata(
    metadata: tuple[_FabricMetadata, ...],
) -> _FabricMetadata | None:
    if not metadata:
        return None

    def rank(item: _FabricMetadata) -> tuple[int, int, int, str]:
        lowered = item.path.casefold()
        container = lowered.split("!/", 1)[0] if "!/" in lowered else ""
        nested_path = lowered.split("!/", 1)[1] if "!/" in lowered else lowered
        is_nested_source = container.endswith(".zip")
        if nested_path == "src/main/resources/fabric.mod.json" and is_nested_source:
            priority = 0
        elif lowered == "src/main/resources/fabric.mod.json":
            priority = 0
        elif nested_path == "fabric.mod.json" and is_nested_source:
            priority = 1
        elif lowered == "fabric.mod.json":
            priority = 1
        elif "!/" not in lowered:
            priority = 2
        else:
            priority = 3
        unresolved_version = int(
            item.version is None
            or "${" in item.version
            or item.version.casefold() in {"unspecified", "unknown"}
        )
        return unresolved_version, priority, len(item.path), item.path

    return min(metadata, key=rank)


def _inspect_embedded_proposal(
    files: list[_ArchiveFile],
    captured: dict[str, _CapturedValue],
    warnings: list[str],
) -> tuple[str | None, str, str | None]:
    candidates = [
        file.relative_path
        for file in files
        if file.relative_path.casefold() == ".minecraft_ai/proposal.approved.json"
        or file.relative_path.casefold().endswith(
            "/.minecraft_ai/proposal.approved.json"
        )
    ]
    if not candidates:
        return None, "absent", None
    if len(candidates) > 1:
        warnings.append("Multiple embedded approved proposals are ambiguous and were not trusted.")
        return None, "ambiguous", None

    proposal_path = candidates[0]
    raw = _captured_bytes(
        captured[proposal_path],
        context=proposal_path,
    )
    if len(raw) > MAX_JSON_BYTES:
        warnings.append("Embedded approved proposal exceeds the JSON parse limit.")
        return proposal_path, "invalid", None
    try:
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top level is not an object")
        proposal = Proposal.from_dict(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        warnings.append(f"Embedded approved proposal is invalid: {exc}")
        return proposal_path, "invalid", None

    if proposal.status is not ProposalStatus.APPROVED:
        warnings.append(
            "Embedded proposal is structurally valid but is not in approved status."
        )
        return proposal_path, "valid_not_approved", proposal.approval_hash or None
    warnings.append(
        "Embedded proposal and approval hash are structurally valid, but the ZIP's "
        "provenance and correspondence to that proposal are not independently attested."
    )
    return proposal_path, "approved_valid", proposal.approval_hash


def _extract_validated_archive(
    archive: zipfile.ZipFile,
    files: list[_ArchiveFile],
    *,
    extract_root: Path,
    archive_stem: str,
    snapshot_hash: str,
    archive_sha256: str,
) -> str:
    if extract_root.exists() and (not extract_root.is_dir() or extract_root.is_symlink()):
        raise ExistingProjectImportError(
            "Extraction root must be a real directory, not a file or symbolic link."
        )
    extract_root.mkdir(parents=True, exist_ok=True)
    resolved_root = extract_root.resolve()
    safe_stem = _SAFE_EXTRACT_NAME.sub("-", archive_stem).strip("._-") or "existing-mod"
    child = resolved_root / f"{safe_stem}-{snapshot_hash.removeprefix('sha256:')[:12]}"
    receipt = (
        resolved_root
        / ".mmm-import-receipts"
        / f"{child.name}.json"
    )
    if child.exists():
        if _completed_extraction_matches(
            child,
            receipt=receipt,
            files=files,
            archive_sha256=archive_sha256,
            snapshot_hash=snapshot_hash,
        ):
            return str(child)
        _preserve_incomplete_path(child)

    staging = resolved_root / f".{child.name}.staging"
    if staging.exists():
        _preserve_incomplete_path(staging)

    required_bytes = sum(file.size for file in files)
    free_bytes = shutil.disk_usage(resolved_root).free
    reserve_bytes = min(1024 * 1024 * 1024, max(64 * 1024 * 1024, required_bytes // 10))
    if required_bytes + reserve_bytes > free_bytes:
        raise ExistingProjectImportError(
            "The validated archive does not fit in the extraction workspace."
        )
    staging.mkdir()

    try:
        for file in files:
            destination = staging.joinpath(*PurePosixPath(file.relative_path).parts)
            resolved_destination = destination.resolve(strict=False)
            if (
                resolved_destination == staging
                or staging not in resolved_destination.parents
            ):
                raise ExistingProjectImportError(
                    f"Extraction path escaped its new directory: {file.relative_path}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            actual_size = 0
            with archive.open(file.info, "r") as source, destination.open("xb") as target:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    actual_size += len(chunk)
                    if actual_size > file.size or (
                        MAX_SINGLE_FILE_BYTES
                        and actual_size > MAX_SINGLE_FILE_BYTES
                    ):
                        raise ExistingProjectImportError(
                            f"Entry changed size while extracting: {file.relative_path}"
                        )
                    digest.update(chunk)
                    target.write(chunk)
            if actual_size != file.size or digest.hexdigest() != file.sha256:
                raise ExistingProjectImportError(
                    f"Entry changed after validation: {file.relative_path}"
                )
    except Exception:
        _preserve_incomplete_path(staging)
        raise

    try:
        staging.replace(child)
        _write_extraction_receipt(
            receipt,
            child=child,
            archive_sha256=archive_sha256,
            snapshot_hash=snapshot_hash,
            file_count=len(files),
        )
    except Exception:
        if staging.exists():
            _preserve_incomplete_path(staging)
        raise
    return str(child)


def _completed_extraction_matches(
    child: Path,
    *,
    receipt: Path,
    files: list[_ArchiveFile],
    archive_sha256: str,
    snapshot_hash: str,
) -> bool:
    if not child.is_dir() or child.is_symlink():
        return False
    if not receipt.is_file() or receipt.is_symlink():
        return False
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    expected_receipt = {
        "schema_version": "mmm/import-extraction-receipt-v1",
        "archive_sha256": archive_sha256,
        "source_snapshot_hash": snapshot_hash,
        "file_count": len(files),
        "destination": str(child),
    }
    if payload != expected_receipt:
        return False
    for file in files:
        path = child.joinpath(*PurePosixPath(file.relative_path).parts)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(child)
        except (OSError, ValueError):
            return False
        if path.is_symlink() or not resolved.is_file() or resolved.is_symlink():
            return False
        try:
            stat_result = resolved.stat()
            with resolved.open("rb") as stream:
                digest = _hash_stream(stream)
        except OSError:
            return False
        if stat_result.st_size != file.size or digest != file.sha256:
            return False
    return True


def _write_extraction_receipt(
    receipt: Path,
    *,
    child: Path,
    archive_sha256: str,
    snapshot_hash: str,
    file_count: int,
) -> None:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(".json.pending")
    if temporary.exists():
        _preserve_incomplete_path(temporary)
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "mmm/import-extraction-receipt-v1",
                "archive_sha256": archive_sha256,
                "source_snapshot_hash": snapshot_hash,
                "file_count": file_count,
                "destination": str(child),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(receipt)


def _preserve_incomplete_path(path: Path) -> Path:
    """Move a run-owned partial path aside without deleting its contents."""

    resolved_parent = path.parent.resolve()
    resolved = path.resolve(strict=False)
    if resolved.parent != resolved_parent:
        raise ExistingProjectImportError(
            "Incomplete extraction path escaped its expected parent."
        )
    index = 1
    while True:
        candidate = resolved_parent / f"{path.name}.incomplete-{index}"
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        index += 1


def _is_gradle_path(path: str) -> bool:
    lowered = path.casefold()
    name = PurePosixPath(lowered).name
    return (
        name
        in {
            "build.gradle",
            "build.gradle.kts",
            "gradle.properties",
            "gradlew",
            "gradlew.bat",
            "settings.gradle",
            "settings.gradle.kts",
        }
        or lowered.startswith("gradle/wrapper/")
    )


def _is_source_path(path: str) -> bool:
    portable = PurePosixPath(path.casefold())
    return portable.suffix in _SOURCE_SUFFIXES and "src" in portable.parts


def _is_mixin_path(path: str) -> bool:
    name = PurePosixPath(path.casefold()).name
    return name.endswith(".mixins.json") or (
        name.startswith("mixin") and name.endswith(".json")
    )


def _is_access_widener_path(path: str) -> bool:
    return PurePosixPath(path.casefold()).suffix == ".accesswidener"


def _is_asset_path(path: str) -> bool:
    return "assets" in PurePosixPath(path.casefold()).parts


def _is_world_path(path: str) -> bool:
    portable = PurePosixPath(path.casefold())
    parts = portable.parts
    return (
        portable.suffix in _WORLD_SUFFIXES
        or "saves" in parts
        or "worlds" in parts
        or (
            len(parts) >= 2
            and parts[0] == ".minecraft_ai"
            and parts[1] == "world"
        )
    )


def _is_mod_jar_path(path: str) -> bool:
    portable = PurePosixPath(path.casefold())
    if portable.suffix != ".jar":
        return False
    return not (
        portable.name == "gradle-wrapper.jar"
        or (
            "gradle" in portable.parts
            and "wrapper" in portable.parts
        )
    )


def _is_release_bundle(path: str) -> bool:
    portable = PurePosixPath(path.casefold())
    return (
        _is_mod_jar_path(path)
        or portable.suffix in {".mrpack", ".zip"}
        or portable.name in {"release-manifest.json", "release_manifest.json"}
    )


def _sorted_paths(*groups: Any) -> tuple[str, ...]:
    flattened: set[str] = set()
    for group in groups:
        if isinstance(group, str):
            flattened.add(group)
            continue
        for value in group:
            if value:
                flattened.add(str(value))
    return tuple(sorted(flattened, key=lambda value: (value.casefold(), value)))


def _plain_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _hash_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value
