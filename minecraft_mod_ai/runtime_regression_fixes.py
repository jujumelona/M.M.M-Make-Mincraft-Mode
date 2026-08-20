from __future__ import annotations

"""Temporary compatibility repairs not yet moved into canonical owners.

Only the remaining ecosystem-discovery gap and research-RAG integration live here.
Do not add new production behavior to this module.
"""

from functools import wraps
from typing import Any


_INSTALLED = False


def _fix_ecosystem_discovery() -> None:
    from . import ecosystem_discovery as discovery

    cls = discovery.EcosystemDiscoveryClient
    current_search = cls.search
    if not getattr(current_search, "_mmm_openverse_route_v1", False):

        @wraps(current_search)
        def search(
            self: Any,
            provider: str,
            query: str,
            *,
            cursor: str = "",
            limit: int = 20,
            minecraft_version: str | None = None,
            loader: str | None = None,
            target_profile: str = "minecraft_mod",
        ) -> dict[str, Any]:
            normalized_provider = provider.strip().lower()
            if normalized_provider != "openverse_images":
                return current_search(
                    self,
                    provider,
                    query,
                    cursor=cursor,
                    limit=limit,
                    minecraft_version=minecraft_version,
                    loader=loader,
                    target_profile=target_profile,
                )

            query = query.strip()
            target_profile = target_profile.strip().lower()
            if target_profile not in discovery._TARGET_PROFILES:
                raise discovery.SpecValidationError(
                    f"Unsupported ecosystem discovery target profile: {target_profile!r}"
                )
            if not query or len(query.encode("utf-8")) > discovery._MAX_QUERY_BYTES:
                raise discovery.SpecValidationError(
                    "Discovery query must be non-empty and within the query byte policy."
                )
            if type(limit) is not int or not 1 <= limit <= discovery._MAX_PAGE_ITEMS:
                raise discovery.SpecValidationError(
                    f"limit must be between 1 and {discovery._MAX_PAGE_ITEMS}."
                )
            target = discovery._normalize_discovery_target(
                minecraft_version,
                loader,
                target_profile=target_profile,
            )
            page = discovery._decode_cursor(
                cursor,
                provider=normalized_provider,
                query=query,
                position_kind="page",
                target_profile=target_profile,
                minecraft_version=target.minecraft_version,
                loader=target.loader,
            ) or 1
            candidates, total, next_position = self._search_openverse(
                query,
                media_type="images",
                page=page,
                limit=limit,
            )
            next_cursor = (
                discovery._encode_cursor(
                    provider=normalized_provider,
                    query=query,
                    position_kind="page",
                    position=next_position,
                    target_profile=target_profile,
                    minecraft_version=target.minecraft_version,
                    loader=target.loader,
                )
                if next_position is not None
                else ""
            )
            payload = {
                "schema_version": "mmm/ecosystem-discovery-page-v2",
                "provider": normalized_provider,
                "query": query,
                "query_sha256": discovery._sha256_text(query),
                "minecraft_version": target.minecraft_version,
                "loader": target.loader,
                "target_exact": target.exact,
                "target_profile": target_profile,
                "candidates": [candidate.to_dict() for candidate in candidates],
                "returned": len(candidates),
                "provider_total_estimate": total,
                "provider_truncated": False,
                "provider_result_limit": None,
                "next_cursor": next_cursor,
                "download_performed": False,
                "authorization": "none",
                "selection_policy": (
                    "Media results remain origin/license hypotheses until their landing "
                    "page, attribution, dimensions, and reuse terms are verified."
                ),
            }
            payload["page_sha256"] = discovery._sha256_text(
                discovery.canonical_json(payload)
            )
            return payload

        search._mmm_openverse_route_v1 = True  # type: ignore[attr-defined]
        cls.search = search

    current_openverse = cls._search_openverse
    if not getattr(current_openverse, "_mmm_openverse_candidates_v1", False):

        def search_openverse(
            self: Any,
            query: str,
            *,
            media_type: str,
            page: int,
            limit: int,
        ):
            raw = self._get_json(
                f"https://api.openverse.org/v1/{media_type}/",
                params={
                    "q": query,
                    "license": "cc0,pdm,by,by-sa",
                    "license_type": "modification",
                    "mature": "false",
                    "page": str(page),
                    "page_size": str(limit),
                },
                provider="openverse",
            )
            if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
                raise discovery.EcosystemDiscoveryUnavailable(
                    "Openverse returned an invalid media search response."
                )
            total = discovery._nonnegative_int(raw.get("result_count"))
            candidates: list[Any] = []
            for item in raw["results"]:
                if not isinstance(item, dict) or item.get("mature") is True:
                    continue
                identifier = str(item.get("id", "")).strip()
                license_name = str(item.get("license", "")).strip().lower()
                license_version = str(item.get("license_version", "")).strip()
                if not identifier or license_name not in {
                    "cc0",
                    "pdm",
                    "by",
                    "by-sa",
                }:
                    continue
                license_id = discovery._creative_commons_id(
                    license_name,
                    license_version,
                )
                source_url = discovery._safe_https_url(
                    item.get("foreign_landing_url") or item.get("detail_url")
                )
                if not source_url:
                    continue
                title = str(item.get("title") or "Untitled").strip() or "Untitled"
                creator = str(item.get("creator") or "").strip()
                attribution = str(item.get("attribution") or "").strip()
                if not attribution:
                    attribution = (
                        f"{title} by {creator}, {license_id}"
                        if creator
                        else f"{title}, {license_id}"
                    )
                stable = {
                    "id": identifier,
                    "title": title,
                    "creator": creator,
                    "source_url": source_url,
                    "license_id": license_id,
                    "provider": str(item.get("provider") or ""),
                    "source": str(item.get("source") or ""),
                }
                candidates.append(
                    discovery.EcosystemCandidate(
                        candidate_id=f"openverse:{identifier}",
                        provider="openverse_images",
                        resource_kind="image_asset",
                        title=title,
                        summary=(
                            f"Openverse image metadata by {creator}."
                            if creator
                            else "Openverse image metadata."
                        ),
                        source_url=source_url,
                        api_url=discovery._safe_https_url(
                            item.get("detail_url"),
                            allow_empty=True,
                        ),
                        license_id=license_id,
                        license_url=discovery._safe_https_url(
                            item.get("license_url"),
                            allow_empty=True,
                        ),
                        license_policy=discovery._media_license_policy(license_name),
                        minecraft_version="not_applicable",
                        loader="not_applicable",
                        compatibility=(
                            "visual_reference_only; verify origin, dimensions, and exact "
                            "license before reuse"
                        ),
                        attribution=attribution,
                        preview_urls=discovery._safe_preview_urls(
                            [item.get("thumbnail")],
                            allowed_hosts=None,
                        ),
                        reuse_status="origin_license_verification_required",
                        evidence_sha256=discovery._sha256_text(
                            discovery.canonical_json(stable)
                        ),
                        metadata={
                            "creator": creator,
                            "provider": stable["provider"],
                            "source": stable["source"],
                        },
                    )
                )
            page_count = discovery._nonnegative_int(raw.get("page_count"))
            return (
                candidates,
                total,
                page + 1 if page_count and page < page_count else None,
            )

        search_openverse._mmm_openverse_candidates_v1 = True  # type: ignore[attr-defined]
        cls._search_openverse = search_openverse

    def seed_query(prompt: str, game_design: dict[str, Any]) -> str:
        parts = [
            prompt,
            str(game_design.get("title", "")),
            str(game_design.get("pitch", "")),
        ]
        for item in game_design.get("modules", []):
            if isinstance(item, dict):
                parts.append(str(item.get("reason") or item.get("name") or ""))
        for item in game_design.get("assets", []):
            if isinstance(item, dict):
                parts.append(str(item.get("brief") or ""))
        return " ".join(part.strip() for part in parts if part.strip())

    seed_query._mmm_lossless_seed_query_v1 = True  # type: ignore[attr-defined]
    discovery._seed_query = seed_query


def _fix_research_hotpaths() -> None:
    from . import centroid_vector_rag, rag_index, research_rag_performance

    research_rag_performance.harden(rag_index, centroid_vector_rag)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _fix_ecosystem_discovery()
    _fix_research_hotpaths()
    _INSTALLED = True


__all__ = ["install"]
