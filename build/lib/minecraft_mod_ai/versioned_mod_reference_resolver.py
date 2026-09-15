from __future__ import annotations

"""Deterministic GitHub resolver for exact-target Minecraft reference revisions."""

import base64
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from .reuse_license import is_reusable_source_license
from .target_contract import minecraft_version_tuple
from .versioned_mod_reference_catalog import ReferenceFamily, exact_version_in_text

_GITHUB = "https://api.github.com"
_METADATA_LEAVES = frozenset(
    {
        "gradle.properties",
        "fabric.mod.json",
        "mods.toml",
        "neoforge.mods.toml",
        "build.gradle",
        "build.gradle.kts",
        "libs.versions.toml",
    }
)
_MC_PROPERTY = re.compile(
    r"(?im)^\s*(?:minecraft_version|minecraftVersion|minecraft\.version|mc_version|mcVersion)\s*=\s*([^\s#]+)"
)
_YARN_PROPERTY = re.compile(
    r"(?im)^\s*(?:yarn_mappings|yarn_version|mappings_version)\s*=\s*([^\s#]+)"
)
_JAVA_PATTERNS = (
    re.compile(r"(?im)^\s*(?:java_version|javaVersion|java)\s*=\s*(\d{1,2})\s*$"),
    re.compile(r"(?i)JavaLanguageVersion\.of\(\s*(\d{1,2})\s*\)"),
    re.compile(r"(?i)(?:sourceCompatibility|targetCompatibility)\s*=\s*(?:JavaVersion\.)?VERSION_(\d{1,2})"),
)
_FABRIC_API_PROPERTY = re.compile(
    r"(?im)^\s*(?:fabric_version|fabric_api_version|fabric-api\.version|fabric_api|fapi_version)\s*=\s*([^\s#]+)"
)
_LOADER_PROPERTY = re.compile(
    r"(?im)^\s*(?:loader_version|fabric_loader_version|fabricloader_version)\s*=\s*([^\s#]+)"
)


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, "").strip() or default)
    except ValueError:
        value = default
    return max(low, min(high, value))


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _version_numbers(value: str) -> tuple[int, ...]:
    head = str(value or "").split("+", 1)[0].split("-", 1)[0]
    return tuple(int(item) for item in re.findall(r"\d+", head)[:4])


def _version_not_newer(donor: str, target: str) -> bool:
    donor_numbers = _version_numbers(donor)
    target_numbers = _version_numbers(target)
    if not donor_numbers or not target_numbers:
        return True
    width = max(len(donor_numbers), len(target_numbers))
    return donor_numbers + (0,) * (width - len(donor_numbers)) <= target_numbers + (0,) * (
        width - len(target_numbers)
    )


def _same_mc_mapping(value: str, minecraft_version: str) -> bool:
    text = str(value or "").strip()
    return not text or text.startswith("+build.") or exact_version_in_text(text, minecraft_version)


def _dependency_mc_suffix_compatible(value: str, minecraft_version: str) -> bool:
    text = str(value or "").strip()
    if "+" not in text:
        return True
    suffix = text.rsplit("+", 1)[-1]
    if re.fullmatch(r"\d+(?:\.\d+){1,2}", suffix):
        return suffix == minecraft_version
    return True


@dataclass(frozen=True)
class CompatibilityProof:
    minecraft_exact: bool
    loader_exact: bool
    mappings_compatible: bool
    java_compatible: bool
    loader_coordinate_compatible: bool
    api_coordinate_compatible: bool

    @property
    def admitted(self) -> bool:
        return all(
            (
                self.minecraft_exact,
                self.loader_exact,
                self.mappings_compatible,
                self.java_compatible,
                self.loader_coordinate_compatible,
                self.api_coordinate_compatible,
            )
        )


@dataclass(frozen=True)
class ResolvedReference:
    family: ReferenceFamily
    ref_name: str
    commit_sha: str
    tree_sha: str
    license_spdx: str
    tree: tuple[dict[str, Any], ...]
    metadata_sha256: str
    compatibility: CompatibilityProof


class ExactReferenceResolver:
    """Resolve curated repositories to exact compatible immutable revisions."""

    def __init__(
        self,
        *,
        minecraft_version: str,
        loader: str,
        mappings: str,
        java_version: int | str,
        fabric_loader: str = "",
        fabric_api: str = "",
    ) -> None:
        self.minecraft_version = str(minecraft_version).strip()
        self.loader = str(loader).strip().casefold()
        self.mappings = str(mappings).strip()
        self.java_version = int(str(java_version).split(".", 1)[0])
        self.fabric_loader = str(fabric_loader or "").strip()
        self.fabric_api = str(fabric_api or "").strip()
        self.ref_probe_limit = _env_int("MMM_REFERENCE_REF_PROBES", 12, 2, 32)
        self.ref_pages = _env_int("MMM_REFERENCE_REF_PAGES", 2, 1, 4)
        token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "MMM-versioned-reference/1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.Client(headers=headers, timeout=httpx.Timeout(12.0, read=18.0))
        self._lock = threading.RLock()
        self._resolved: dict[str, ResolvedReference | None] = {}
        self._refs_cache: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
        self._commit_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._license_cache: dict[tuple[str, str], str] = {}
        self._blob_cache: dict[tuple[str, str], str] = {}

    def close(self) -> None:
        self._client.close()

    def resolve(self, family: ReferenceFamily) -> ResolvedReference | None:
        cache_key = f"{family.repository}|{self.minecraft_version}|{self.loader}|{self.mappings}"
        with self._lock:
            cached = self._resolved.get(cache_key, ...)
        if cached is not ...:
            return cached
        try:
            repo = self._json(f"{_GITHUB}/repos/{family.repository}")
            if not isinstance(repo, dict):
                return None
            default_branch = str(repo.get("default_branch") or "").strip()
            refs = self._refs(family.repository, default_branch)
            refs.sort(
                key=lambda item: self._ref_score(item[0], default_branch),
                reverse=True,
            )
            resolved = None
            for ref_name, sha in refs[: self.ref_probe_limit]:
                try:
                    candidate = self._inspect(
                        family,
                        ref_name=ref_name,
                        ref_sha=sha,
                    )
                except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if candidate is not None:
                    resolved = candidate
                    break
        except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None
        with self._lock:
            self._resolved[cache_key] = resolved
        return resolved

    def blob(self, repository: str, sha: str) -> str:
        key = (repository, sha)
        with self._lock:
            cached = self._blob_cache.get(key)
        if cached is not None:
            return cached
        payload = self._json(f"{_GITHUB}/repos/{repository}/git/blobs/{sha}")
        if not isinstance(payload, dict):
            return ""
        content = str(payload.get("content") or "")
        if str(payload.get("encoding") or "").casefold() == "base64":
            try:
                decoded = base64.b64decode(content, validate=False).decode(
                    "utf-8", errors="replace"
                )
            except (ValueError, UnicodeError):
                decoded = ""
        else:
            decoded = content
        if len(decoded.encode("utf-8")) <= 512 * 1024:
            with self._lock:
                self._blob_cache[key] = decoded
        return decoded

    def _json(self, url: str) -> Any:
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def _refs(self, repository: str, default_branch: str) -> list[tuple[str, str]]:
        cache_key = (repository, default_branch)
        with self._lock:
            cached = self._refs_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        refs: list[tuple[str, str]] = []
        for endpoint in ("branches", "tags"):
            for page in range(1, self.ref_pages + 1):
                payload = self._json(
                    f"{_GITHUB}/repos/{repository}/{endpoint}?per_page=100&page={page}"
                )
                if not isinstance(payload, list):
                    break
                for row in payload:
                    if not isinstance(row, dict) or not isinstance(row.get("commit"), dict):
                        continue
                    name = str(row.get("name") or "").strip()
                    sha = str(row["commit"].get("sha") or "").strip()
                    if name and sha:
                        refs.append((name, sha))
                if len(payload) < 100:
                    break

        if default_branch and all(name != default_branch for name, _sha in refs):
            branch = self._json(
                f"{_GITHUB}/repos/{repository}/branches/{quote(default_branch, safe='')}"
            )
            if isinstance(branch, dict) and isinstance(branch.get("commit"), dict):
                sha = str(branch["commit"].get("sha") or "").strip()
                if sha:
                    refs.append((default_branch, sha))

        unique: dict[str, str] = {}
        for name, sha in refs:
            unique.setdefault(name, sha)
        result = tuple(unique.items())
        with self._lock:
            self._refs_cache[cache_key] = result
        return list(result)

    def _ref_score(self, name: str, default_branch: str) -> tuple[int, int, str]:
        exact = exact_version_in_text(name, self.minecraft_version)
        version = minecraft_version_tuple(self.minecraft_version)
        major_minor = ".".join(str(item) for item in version[:2])
        broad = exact_version_in_text(name, major_minor) if major_minor else False
        default = name == default_branch
        return (3 if exact else 2 if broad else 1 if default else 0, int(default), name)

    def _commit_object(self, repository: str, ref_sha: str) -> dict[str, Any] | None:
        current_sha = str(ref_sha or "").strip()
        if not current_sha:
            return None
        seen: set[str] = set()
        for _depth in range(4):
            key = (repository, current_sha)
            with self._lock:
                cached = self._commit_cache.get(key)
            if cached is not None:
                return cached
            if current_sha in seen:
                return None
            seen.add(current_sha)
            try:
                commit = self._json(
                    f"{_GITHUB}/repos/{repository}/git/commits/{quote(current_sha, safe='')}"
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
                tag = self._json(
                    f"{_GITHUB}/repos/{repository}/git/tags/{quote(current_sha, safe='')}"
                )
                if not isinstance(tag, dict) or not isinstance(tag.get("object"), dict):
                    return None
                target = tag["object"]
                target_type = str(target.get("type") or "").strip().casefold()
                target_sha = str(target.get("sha") or "").strip()
                if target_type not in {"commit", "tag"} or not target_sha:
                    return None
                current_sha = target_sha
                continue
            if not isinstance(commit, dict) or not isinstance(commit.get("tree"), dict):
                return None
            immutable_sha = str(commit.get("sha") or current_sha).strip()
            if not immutable_sha:
                return None
            with self._lock:
                self._commit_cache[(repository, immutable_sha)] = commit
                self._commit_cache[(repository, ref_sha)] = commit
            return commit
        return None

    def _license_at_commit(self, repository: str, commit_sha: str) -> str:
        key = (repository, commit_sha)
        with self._lock:
            cached = self._license_cache.get(key)
        if cached is not None:
            return cached
        payload = self._json(
            f"{_GITHUB}/repos/{repository}/license?ref={quote(commit_sha, safe='')}"
        )
        license_info = payload.get("license") if isinstance(payload, dict) else None
        spdx = (
            str(license_info.get("spdx_id") or "").strip()
            if isinstance(license_info, dict)
            else ""
        )
        with self._lock:
            self._license_cache[key] = spdx
        return spdx

    def _inspect(
        self,
        family: ReferenceFamily,
        *,
        ref_name: str,
        ref_sha: str,
    ) -> ResolvedReference | None:
        commit = self._commit_object(family.repository, ref_sha)
        if not isinstance(commit, dict) or not isinstance(commit.get("tree"), dict):
            return None
        immutable_sha = str(commit.get("sha") or "").strip()
        tree_sha = str(commit["tree"].get("sha") or "").strip()
        if not immutable_sha or not tree_sha:
            return None
        tree_payload = self._json(
            f"{_GITHUB}/repos/{family.repository}/git/trees/{tree_sha}?recursive=1"
        )
        raw_tree = tree_payload.get("tree") if isinstance(tree_payload, dict) else None
        if not isinstance(raw_tree, list):
            return None
        tree = tuple(row for row in raw_tree if isinstance(row, dict))
        metadata_rows = [
            row
            for row in tree
            if str(row.get("type") or "") == "blob"
            and str(row.get("path") or "").rsplit("/", 1)[-1] in _METADATA_LEAVES
        ]
        metadata_rows.sort(
            key=lambda row: (str(row.get("path") or "").count("/"), str(row.get("path") or ""))
        )
        metadata_parts: list[str] = []
        for row in metadata_rows[:10]:
            sha = str(row.get("sha") or "")
            path = str(row.get("path") or "")
            if sha:
                metadata_parts.append(f"### {path}\n{self.blob(family.repository, sha)}")
        metadata = "\n".join(metadata_parts)
        if not metadata:
            return None
        proof = self._compatibility(metadata, family)
        if not proof.admitted:
            return None
        license_spdx = self._license_at_commit(family.repository, immutable_sha)
        if not is_reusable_source_license(license_spdx):
            return None
        return ResolvedReference(
            family=family,
            ref_name=ref_name,
            commit_sha=immutable_sha,
            tree_sha=tree_sha,
            license_spdx=license_spdx,
            tree=tree,
            metadata_sha256=_sha256(metadata),
            compatibility=proof,
        )

    def _compatibility(self, metadata: str, family: ReferenceFamily) -> CompatibilityProof:
        mc_values = [match.group(1).strip("\"'") for match in _MC_PROPERTY.finditer(metadata)]
        mc_exact = any(value == self.minecraft_version for value in mc_values)
        if not mc_exact:
            mc_exact = any(
                "minecraft" in line.casefold()
                and exact_version_in_text(line, self.minecraft_version)
                for line in metadata.splitlines()
            )

        loader_exact = self.loader in family.loaders
        lower = metadata.casefold()
        if self.loader == "fabric":
            loader_exact = loader_exact and (
                "fabric.mod.json" in lower or "fabric-loader" in lower or "fabricloader" in lower
            )
        elif self.loader == "neoforge":
            loader_exact = loader_exact and "neoforge" in lower
        elif self.loader == "forge":
            loader_exact = loader_exact and "forge" in lower

        mapping_values = [match.group(1).strip("\"'") for match in _YARN_PROPERTY.finditer(metadata)]
        mappings_ok = all(
            _same_mc_mapping(value, self.minecraft_version) for value in mapping_values
        )
        if self.mappings and mapping_values:
            mappings_ok = mappings_ok and all(
                value.startswith("+build.")
                or value == self.mappings
                or exact_version_in_text(value, self.minecraft_version)
                for value in mapping_values
            )

        donor_java = [
            int(match.group(1))
            for pattern in _JAVA_PATTERNS
            for match in pattern.finditer(metadata)
        ]
        java_ok = not donor_java or max(donor_java) <= self.java_version

        loader_values = [
            match.group(1).strip("\"'") for match in _LOADER_PROPERTY.finditer(metadata)
        ]
        loader_coordinate_ok = True
        if self.loader == "fabric" and self.fabric_loader and loader_values:
            loader_coordinate_ok = all(
                _version_not_newer(value, self.fabric_loader) for value in loader_values
            )

        api_values = [
            match.group(1).strip("\"'") for match in _FABRIC_API_PROPERTY.finditer(metadata)
        ]
        api_coordinate_ok = all(
            _dependency_mc_suffix_compatible(value, self.minecraft_version) for value in api_values
        )
        if self.loader == "fabric" and self.fabric_api and api_values:
            api_coordinate_ok = api_coordinate_ok and all(
                _version_not_newer(value, self.fabric_api) for value in api_values
            )

        return CompatibilityProof(
            minecraft_exact=mc_exact,
            loader_exact=loader_exact,
            mappings_compatible=mappings_ok,
            java_compatible=java_ok,
            loader_coordinate_compatible=loader_coordinate_ok,
            api_coordinate_compatible=api_coordinate_ok,
        )


__all__ = ["CompatibilityProof", "ExactReferenceResolver", "ResolvedReference"]
