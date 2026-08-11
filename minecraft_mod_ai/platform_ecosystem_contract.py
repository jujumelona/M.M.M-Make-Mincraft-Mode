from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import replace
from functools import wraps
from typing import Any

from .platform_catalog import adapter_for_target


_ACTIVE_TARGET: ContextVar[tuple[str, str] | None] = ContextVar(
    "mmm_ecosystem_target", default=None
)


def install(ecosystem_module: Any, complete_planner_module: Any) -> None:
    client_cls = ecosystem_module.EcosystemDiscoveryClient
    if getattr(client_cls.search, "_mmm_dynamic_platform_target", False):
        complete_planner_module.discover_seed_bundle = ecosystem_module.discover_seed_bundle
        return

    original_search = client_cls.search
    original_modrinth = client_cls._search_modrinth
    original_github = client_cls._search_github
    original_inspect = client_cls.inspect_modrinth_project
    original_seed = ecosystem_module.discover_seed_bundle

    def selected_target(
        minecraft_version: str,
        loader: str,
    ) -> tuple[str, str]:
        active = _ACTIVE_TARGET.get()
        version, selected_loader = active or (minecraft_version, loader)
        try:
            adapter_for_target(version, selected_loader)
        except ValueError as exc:
            raise ecosystem_module.SpecValidationError(str(exc)) from exc
        return version, selected_loader

    @wraps(original_search)
    def search(
        self: Any,
        provider: str,
        query: str,
        *,
        cursor: str = "",
        limit: int = 20,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
        target_profile: str = "minecraft_mod",
    ) -> dict[str, Any]:
        version, selected_loader = selected_target(minecraft_version, loader)
        token = _ACTIVE_TARGET.set((version, selected_loader))
        try:
            # The legacy public search has an obsolete 1.20.1 guard. Internal provider
            # methods below consume the ContextVar target, then the returned page is
            # rebound to that same target before hashing.
            page = original_search(
                self,
                provider,
                query,
                cursor=cursor,
                limit=limit,
                minecraft_version="1.20.1",
                loader="fabric",
                target_profile=target_profile,
            )
        finally:
            _ACTIVE_TARGET.reset(token)
        page = dict(page)
        page["minecraft_version"] = version
        page["loader"] = selected_loader
        page["page_sha256"] = ecosystem_module._sha256_text(
            ecosystem_module.canonical_json(
                {key: value for key, value in page.items() if key != "page_sha256"}
            )
        )
        return page

    @wraps(original_modrinth)
    def search_modrinth(self: Any, query: str, *, offset: int, limit: int):
        active = _ACTIVE_TARGET.get()
        if active is None or active == ("1.20.1", "fabric"):
            return original_modrinth(self, query, offset=offset, limit=limit)
        version, loader = active
        url = "https://api.modrinth.com/v2/search"
        raw = self._get_json(
            url,
            params={
                "query": query,
                "facets": json.dumps(
                    [
                        [f"versions:{version}"],
                        [f"categories:{loader}"],
                        ["project_type:mod"],
                        ["open_source:true"],
                    ],
                    separators=(",", ":"),
                ),
                "index": "relevance",
                "offset": str(offset),
                "limit": str(limit),
            },
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("hits"), list):
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                "Modrinth returned an invalid search response."
            )
        total = ecosystem_module._nonnegative_int(raw.get("total_hits"))
        candidates: list[Any] = []
        for hit in raw["hits"]:
            if not isinstance(hit, dict):
                continue
            project_id = str(hit.get("project_id", "")).strip()
            slug = str(hit.get("slug", "")).strip()
            license_id = str(hit.get("license", "")).strip()
            if not project_id or not slug or not license_id:
                continue
            project_type = str(hit.get("project_type", "mod"))
            source_url = f"https://modrinth.com/{project_type}/{slug}"
            stable = {
                "project_id": project_id,
                "slug": slug,
                "title": str(hit.get("title", "")),
                "description": str(hit.get("description", "")),
                "license": license_id,
                "versions": sorted(str(v) for v in hit.get("versions", [])),
                "categories": sorted(str(v) for v in hit.get("categories", [])),
                "selected_target": {"minecraft_version": version, "loader": loader},
            }
            candidates.append(
                ecosystem_module.EcosystemCandidate(
                    candidate_id=f"modrinth:{project_id}",
                    provider="modrinth",
                    resource_kind=project_type,
                    title=stable["title"],
                    summary=stable["description"],
                    source_url=source_url,
                    api_url=f"https://api.modrinth.com/v2/project/{project_id}",
                    license_id=license_id,
                    license_url="",
                    license_policy=ecosystem_module._code_license_policy(license_id),
                    minecraft_version=version,
                    loader=loader,
                    compatibility="search_metadata_exact; version_file_inspection_required",
                    attribution="",
                    preview_urls=ecosystem_module._safe_preview_urls(
                        [hit.get("icon_url"), *hit.get("gallery", [])],
                        allowed_hosts={"cdn.modrinth.com"},
                    ),
                    reuse_status="candidate_only_not_downloaded",
                    evidence_sha256=ecosystem_module._sha256_text(
                        ecosystem_module.canonical_json(stable)
                    ),
                )
            )
        next_offset = offset + limit if offset + limit < total else None
        return candidates, total, next_offset

    @wraps(original_github)
    def search_github(
        self: Any,
        query: str,
        *,
        page: int,
        limit: int,
        target_profile: str = "minecraft_mod",
    ):
        candidates, total, next_page = original_github(
            self,
            query,
            page=page,
            limit=limit,
            target_profile=target_profile,
        )
        active = _ACTIVE_TARGET.get()
        if active is None or target_profile != "minecraft_mod":
            return candidates, total, next_page
        version, loader = active
        return (
            [replace(item, minecraft_version=version, loader=loader) for item in candidates],
            total,
            next_page,
        )

    @wraps(original_inspect)
    def inspect_modrinth_project(
        self: Any,
        project_id: str,
        *,
        minecraft_version: str = "1.20.1",
        loader: str = "fabric",
    ) -> dict[str, Any]:
        version, selected_loader = selected_target(minecraft_version, loader)
        if version == "1.20.1" and selected_loader == "fabric":
            return original_inspect(
                self,
                project_id,
                minecraft_version=version,
                loader=selected_loader,
            )
        if not ecosystem_module._MODRINTH_ID.fullmatch(project_id):
            raise ecosystem_module.SpecValidationError("Invalid Modrinth project ID or slug.")
        project_url = f"https://api.modrinth.com/v2/project/{project_id}"
        project = self._get_json(project_url)
        versions = self._get_json(
            project_url + "/version",
            params={
                "loaders": json.dumps([selected_loader]),
                "game_versions": json.dumps([version]),
                "include_changelog": "false",
            },
        )
        if not isinstance(project, dict) or not isinstance(versions, list):
            raise ecosystem_module.EcosystemDiscoveryUnavailable(
                "Modrinth returned an invalid project inspection response."
            )
        license_value = project.get("license")
        if isinstance(license_value, dict):
            license_id = str(license_value.get("id", "")).strip()
            license_url = str(license_value.get("url", "") or "").strip()
        else:
            license_id = str(license_value or "").strip()
            license_url = ""
        normalized_versions = [
            ecosystem_module._normalize_modrinth_version(
                item,
                minecraft_version=version,
                loader=selected_loader,
            )
            for item in versions
            if isinstance(item, dict)
        ]
        compatible = [item for item in normalized_versions if item["eligible_for_selection"]]
        payload = {
            "schema_version": "mmm/modrinth-inspection-v1",
            "project_id": str(project.get("id", project_id)),
            "slug": str(project.get("slug", project_id)),
            "title": str(project.get("title", "")),
            "license_id": license_id,
            "license_url": ecosystem_module._safe_https_url(license_url, allow_empty=True),
            "license_policy": ecosystem_module._code_license_policy(license_id),
            "minecraft_version": version,
            "loader": selected_loader,
            "versions": normalized_versions,
            "exact_compatible_version_found": bool(compatible),
            "eligible_version_ids": [item["version_id"] for item in compatible],
            "compatibility_gate": (
                "candidate_requires_dependency_closure_and_verified_download"
                if compatible
                else "blocked_no_exact_version_with_one_primary_strong_digest_file"
            ),
            "download_performed": False,
            "required_next_gate": (
                "Select one listed version, resolve required/incompatible dependencies, "
                "download only after approval, and verify the advertised SHA-512."
            ),
        }
        payload["inspection_sha256"] = ecosystem_module._sha256_text(
            ecosystem_module.canonical_json(payload)
        )
        return payload

    @wraps(original_seed)
    def discover_seed_bundle(
        prompt: str,
        game_design: dict[str, Any],
        *,
        research_brief: dict[str, Any] | None = None,
        client: Any = None,
        route_cursor: str = "",
        route_limit: int = 12,
    ) -> dict[str, Any]:
        target = None
        if isinstance(research_brief, dict):
            raw = research_brief.get("_mmm_platform_target")
            if isinstance(raw, dict):
                version = str(raw.get("minecraft_version", ""))
                loader = str(raw.get("loader", "fabric"))
                if version:
                    adapter_for_target(version, loader)
                    target = (version, loader)
        token = _ACTIVE_TARGET.set(target)
        try:
            return original_seed(
                prompt,
                game_design,
                research_brief=research_brief,
                client=client,
                route_cursor=route_cursor,
                route_limit=route_limit,
            )
        finally:
            _ACTIVE_TARGET.reset(token)

    search._mmm_dynamic_platform_target = True
    search_modrinth._mmm_dynamic_platform_target = True
    search_github._mmm_dynamic_platform_target = True
    inspect_modrinth_project._mmm_dynamic_platform_target = True
    discover_seed_bundle._mmm_dynamic_platform_target = True
    client_cls.search = search
    client_cls._search_modrinth = search_modrinth
    client_cls._search_github = search_github
    client_cls.inspect_modrinth_project = inspect_modrinth_project
    ecosystem_module.discover_seed_bundle = discover_seed_bundle
    complete_planner_module.discover_seed_bundle = discover_seed_bundle
