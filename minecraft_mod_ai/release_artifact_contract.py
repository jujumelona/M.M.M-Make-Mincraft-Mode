from __future__ import annotations

"""Verified release-artifact rewrite and stale-target handling."""

import json
import os
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .complete_orchestrator_support import CompleteProductionError, file_sha256


def stable_payload_sha256(value: Any) -> str:
    import hashlib

    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def generation_receipt_sort_key(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, dict):
        return ("", "", stable_payload_sha256(value))
    owner = next(
        (
            str(value.get(key) or "")
            for key in ("module_id", "entity_id", "pack_id", "sound_id")
            if value.get(key)
        ),
        "",
    )
    return (
        owner,
        str(value.get("schema_version") or ""),
        stable_payload_sha256(value),
    )


def replace_stale_file_target(
    target: Path,
    action: Callable[[], Any],
) -> Any:
    path = target.expanduser().resolve()
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise CompleteProductionError(
                f"Stale package target is not a regular file: {path}"
            )
        path.unlink()
    return action()


def replace_stale_directory_target(
    target: Path,
    action: Callable[[], Any],
) -> Any:
    path = target.expanduser().resolve()
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise CompleteProductionError(
                f"Stale package target is not a regular directory: {path}"
            )
        shutil.rmtree(path)
    return action()


def attach_verified_release_artifact(
    release_result: dict[str, Any],
    descriptor: dict[str, Any] | None,
    *,
    archive_name: str,
    allowed_root: Path,
    manifest_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    if descriptor is None and not manifest_provenance:
        return dict(release_result)
    if (
        not isinstance(release_result, dict)
        or release_result.get("status") != "PACKAGED"
        or not isinstance(release_result.get("release_zip"), str)
        or not isinstance(release_result.get("sha256"), str)
    ):
        raise CompleteProductionError(
            "Release package receipt is invalid before attachment."
        )
    if descriptor is not None and (
        not archive_name or Path(archive_name).name != archive_name
    ):
        raise CompleteProductionError("Release attachment name is unsafe.")

    root = allowed_root.expanduser().resolve()
    release_path = Path(str(release_result["release_zip"])).expanduser().resolve()
    source = (
        Path(str(descriptor.get("path") or "")).expanduser().resolve()
        if descriptor is not None
        else None
    )
    candidates = [(release_path, "release ZIP")]
    if source is not None:
        candidates.append((source, "release attachment"))
    for candidate, label in candidates:
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CompleteProductionError(
                f"{label} escaped the run workspace: {candidate}"
            ) from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise CompleteProductionError(
                f"{label} is missing or unsafe: {candidate}"
            )

    if file_sha256(release_path) != release_result["sha256"]:
        raise CompleteProductionError(
            "Release ZIP changed before verified attachment."
        )
    expected = (
        str(descriptor.get("sha256") or "")
        if descriptor is not None
        else ""
    )
    if source is not None and (
        not expected or file_sha256(source) != expected
    ):
        raise CompleteProductionError("Release attachment digest mismatch.")

    temp = release_path.with_name("." + release_path.name + ".rewrite.tmp")
    if temp.exists():
        if temp.is_symlink() or not temp.is_file():
            raise CompleteProductionError(
                "Release rewrite temporary target is unsafe."
            )
        temp.unlink()

    member_name = "additional/" + archive_name if descriptor is not None else ""
    try:
        with zipfile.ZipFile(release_path, "r") as original:
            infos = original.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise CompleteProductionError(
                    "Release ZIP contains duplicate member names."
                )
            if "release-manifest.json" not in names:
                raise CompleteProductionError(
                    "Release ZIP has no release manifest."
                )
            if descriptor is not None and member_name in names:
                raise CompleteProductionError(
                    f"Release ZIP already contains attachment member: {member_name}"
                )
            try:
                manifest = json.loads(
                    original.read("release-manifest.json").decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
                raise CompleteProductionError(
                    "Release manifest is unreadable before attachment."
                ) from exc
            if not isinstance(manifest, dict):
                raise CompleteProductionError(
                    "Release manifest is not a JSON object."
                )

            if manifest_provenance:
                for key, value in manifest_provenance.items():
                    if (
                        not isinstance(key, str)
                        or not key
                        or not isinstance(value, str)
                    ):
                        raise CompleteProductionError(
                            "Release manifest provenance must contain "
                            "non-empty string keys and string values."
                        )
                    manifest[key] = value

            if descriptor is not None:
                additional = manifest.get("additional_artifacts")
                if additional is None:
                    additional_map: dict[str, str] = {}
                elif isinstance(additional, dict):
                    additional_map = {
                        str(key): str(value)
                        for key, value in additional.items()
                    }
                else:
                    raise CompleteProductionError(
                        "Release manifest additional_artifacts is invalid."
                    )
                additional_map[archive_name] = expected
                manifest["additional_artifacts"] = dict(
                    sorted(additional_map.items())
                )

            with zipfile.ZipFile(temp, "w") as rewritten:
                manifest_info: zipfile.ZipInfo | None = None
                for info in infos:
                    if info.filename == "release-manifest.json":
                        manifest_info = info
                        continue
                    rewritten.writestr(info, original.read(info.filename))
                if source is not None:
                    attachment_info = zipfile.ZipInfo(
                        member_name,
                        date_time=(1980, 1, 1, 0, 0, 0),
                    )
                    attachment_info.compress_type = zipfile.ZIP_DEFLATED
                    attachment_info.external_attr = (0o644 & 0xFFFF) << 16
                    rewritten.writestr(attachment_info, source.read_bytes())
                if manifest_info is None:
                    raise CompleteProductionError(
                        "Release manifest metadata disappeared."
                    )
                rewritten.writestr(
                    manifest_info,
                    json.dumps(
                        manifest,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
        os.replace(temp, release_path)
    finally:
        if temp.exists():
            temp.unlink()

    updated = dict(release_result)
    updated["sha256"] = file_sha256(release_path)
    if descriptor is not None:
        updated["additional_artifacts"] = dict(
            sorted(
                {
                    **(
                        release_result.get("additional_artifacts")
                        if isinstance(
                            release_result.get("additional_artifacts"),
                            dict,
                        )
                        else {}
                    ),
                    archive_name: expected,
                }.items()
            )
        )
    if manifest_provenance:
        updated["manifest_provenance"] = dict(manifest_provenance)
    return updated


__all__ = [
    "attach_verified_release_artifact",
    "generation_receipt_sort_key",
    "replace_stale_directory_target",
    "replace_stale_file_target",
    "stable_payload_sha256",
]
