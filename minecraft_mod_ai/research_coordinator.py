from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Callable, Mapping

from .ecosystem_discovery import EcosystemDiscoveryClient, discover_seed_bundle
from .spec import SpecValidationError, canonical_json
from .technology_radar import build_technology_radar

TechnologyPageBuilder = Callable[..., dict[str, Any]]
EcosystemPageBuilder = Callable[..., dict[str, Any]]


def collect_technology_radar(
    prompt: str,
    research_brief: Mapping[str, Any] | None = None,
    *,
    page_size: int = 50,
    target: Any = None,
    page_builder: TechnologyPageBuilder = build_technology_radar,
) -> dict[str, Any]:
    """Collect every technology page without imposing a coordinator-wide cap."""
    if type(page_size) is not int or not 1 <= page_size <= 100:
        raise SpecValidationError("Technology page_size must be between 1 and 100.")
    if target is None and isinstance(research_brief, Mapping):
        selected_target = research_brief.get("_mmm_platform_target")
        if isinstance(selected_target, Mapping):
            target = dict(selected_target)

    cursor = ""
    seen_cursors: set[str] = set()
    requirements: list[dict[str, Any]] = []
    requirement_ids: set[str] = set()
    page_receipts: list[dict[str, Any]] = []
    first_page: dict[str, Any] | None = None
    source_sha256 = ""
    total_requirements: int | None = None
    expected_offset = 0

    while True:
        kwargs: dict[str, Any] = {"page_size": page_size, "cursor": cursor}
        if target is not None:
            kwargs["target"] = target
        page = page_builder(prompt, research_brief, **kwargs)
        if not isinstance(page, dict):
            raise SpecValidationError("Technology page must be an object.")
        if page.get("schema_version") != "mmm/technology-radar-page-v1":
            raise SpecValidationError("Unsupported technology radar page schema.")

        pagination = page.get("pagination")
        raw_requirements = page.get("requirements")
        if not isinstance(pagination, dict) or not isinstance(raw_requirements, list):
            raise SpecValidationError(
                "Technology page requires pagination and requirements."
            )
        offset = _strict_non_negative_int(
            pagination.get("offset"), "Technology pagination.offset"
        )
        returned = _strict_non_negative_int(
            pagination.get("returned"), "Technology pagination.returned"
        )
        declared_page_size = _strict_non_negative_int(
            pagination.get("page_size"), "Technology pagination.page_size"
        )
        declared_total = _strict_non_negative_int(
            pagination.get("total_requirements"),
            "Technology pagination.total_requirements",
        )
        next_cursor = pagination.get("next_cursor")
        if not isinstance(next_cursor, str):
            raise SpecValidationError(
                "Technology pagination.next_cursor must be a string."
            )
        if declared_page_size != page_size:
            raise SpecValidationError("Technology page changed the requested page size.")
        if offset != expected_offset:
            raise SpecValidationError(
                "Technology pagination did not advance monotonically."
            )
        if returned != len(raw_requirements):
            raise SpecValidationError(
                "Technology pagination.returned does not match requirements."
            )
        if returned > page_size:
            raise SpecValidationError("Technology page exceeded its per-page limit.")
        if offset + returned > declared_total:
            raise SpecValidationError("Technology page exceeds its declared total.")

        current_source = page.get("source_sha256")
        if not isinstance(current_source, str) or not current_source:
            raise SpecValidationError("Technology page source_sha256 is required.")
        if first_page is None:
            first_page = deepcopy(page)
            source_sha256 = current_source
            total_requirements = declared_total
        else:
            if current_source != source_sha256:
                raise SpecValidationError("Technology pages changed source_sha256.")
            if declared_total != total_requirements:
                raise SpecValidationError(
                    "Technology pages changed their declared total."
                )
            _require_equal_page_invariants(first_page, page)

        for raw_requirement in raw_requirements:
            if not isinstance(raw_requirement, dict):
                raise SpecValidationError("Technology requirements must be objects.")
            requirement_id = raw_requirement.get("requirement_id")
            if not isinstance(requirement_id, str) or not requirement_id:
                raise SpecValidationError("Technology requirement_id is required.")
            if requirement_id in requirement_ids:
                raise SpecValidationError(
                    f"Technology pagination repeated requirement: {requirement_id}"
                )
            requirement_ids.add(requirement_id)
            requirements.append(deepcopy(raw_requirement))

        page_receipts.append(
            {
                "offset": offset,
                "returned": returned,
                "radar_sha256": str(page.get("radar_sha256", "")),
            }
        )
        next_offset = offset + returned
        if not next_cursor:
            if next_offset != declared_total:
                raise SpecValidationError(
                    "Technology pagination ended before its declared total."
                )
            break
        if next_offset <= offset:
            raise SpecValidationError("Technology pagination cursor did not advance.")
        if next_offset >= declared_total:
            raise SpecValidationError(
                "Technology pagination returned a cursor after its declared total."
            )
        if next_cursor == cursor or next_cursor in seen_cursors:
            raise SpecValidationError("Technology pagination repeated a cursor.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        expected_offset = next_offset

    assert first_page is not None and total_requirements is not None
    aggregate = first_page
    aggregate.pop("radar_sha256", None)
    aggregate["aggregate_schema_version"] = "mmm/technology-radar-aggregate-v1"
    aggregate["requirements"] = requirements
    aggregate["pagination"] = {
        "offset": 0,
        "page_size": page_size,
        "returned": len(requirements),
        "total_requirements": total_requirements,
        "next_cursor": "",
        "pages_collected": len(page_receipts),
        "complete": True,
    }
    aggregate["collection_receipt"] = {
        "schema_version": "mmm/technology-page-collection-receipt-v1",
        "page_count": len(page_receipts),
        "page_size": page_size,
        "pages_sha256": _sha256(page_receipts),
    }
    aggregate["radar_sha256"] = _sha256(aggregate)
    return aggregate


def collect_ecosystem_seed_bundle(
    prompt: str,
    game_design: dict[str, Any],
    *,
    research_brief: dict[str, Any] | None = None,
    client: EcosystemDiscoveryClient | None = None,
    route_limit: int = 12,
    page_builder: EcosystemPageBuilder = discover_seed_bundle,
    planning_seed_only: bool = False,
) -> dict[str, Any]:
    """Collect ecosystem route pages, or exactly one bounded planning seed page."""
    if type(route_limit) is not int or not 1 <= route_limit <= 100:
        raise SpecValidationError("Ecosystem route_limit must be between 1 and 100.")
    cursor = ""
    seen_cursors: set[str] = set()
    pages: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    route_receipts: list[dict[str, Any]] = []
    statuses: list[str] = []
    first_page: dict[str, Any] | None = None
    route_sha256 = ""
    query_sha256 = ""
    route_count: int | None = None
    expected_offset: int | None = 0
    previous_offset = -1
    candidate_count = 0
    processed_route_count = 0

    while True:
        page = page_builder(
            prompt,
            game_design,
            research_brief=research_brief,
            client=client,
            route_cursor=cursor,
            route_limit=route_limit,
        )
        if not isinstance(page, dict):
            raise SpecValidationError("Ecosystem route page must be an object.")
        if page.get("schema_version") not in {
            "mmm/ecosystem-seed-bundle-v1",
            "mmm/ecosystem-seed-bundle-v2",
        }:
            raise SpecValidationError("Unsupported ecosystem seed page schema.")

        offset = _strict_non_negative_int(
            page.get("route_offset"), "Ecosystem route_offset"
        )
        declared_count = _strict_non_negative_int(
            page.get("route_count"), "Ecosystem route_count"
        )
        processed = _strict_non_negative_int(
            page.get("processed_route_count"), "Ecosystem processed_route_count"
        )
        remaining = _strict_non_negative_int(
            page.get("remaining_route_count"), "Ecosystem remaining_route_count"
        )
        declared_candidates = _strict_non_negative_int(
            page.get("candidate_count"), "Ecosystem candidate_count"
        )
        next_cursor = page.get("next_route_cursor")
        complete = page.get("routes_complete")
        raw_pages = page.get("pages")
        raw_errors = page.get("errors")
        if not isinstance(next_cursor, str) or type(complete) is not bool:
            raise SpecValidationError(
                "Ecosystem route page requires boolean completion and a string cursor."
            )
        if not isinstance(raw_pages, list) or not isinstance(raw_errors, list):
            raise SpecValidationError("Ecosystem pages and errors must be lists.")
        if expected_offset is None:
            if offset <= previous_offset or offset - previous_offset > route_limit:
                raise SpecValidationError(
                    "Ecosystem route pagination did not advance."
                )
        elif offset != expected_offset:
            raise SpecValidationError("Ecosystem route pagination did not advance.")
        if remaining > declared_count:
            raise SpecValidationError("Ecosystem remaining routes exceed route_count.")
        if offset > declared_count:
            raise SpecValidationError("Ecosystem route_offset exceeds route_count.")

        status = str(page.get("status", ""))
        disabled_page = status == "disabled"
        next_offset = None if disabled_page else declared_count - remaining
        if next_offset is not None:
            if next_offset < offset or next_offset - offset > route_limit:
                raise SpecValidationError(
                    "Ecosystem route page violated its per-page route limit."
                )
            if processed > next_offset - offset:
                raise SpecValidationError(
                    "Ecosystem processed routes exceed this route page."
                )
        elif processed:
            raise SpecValidationError(
                "Disabled ecosystem discovery may not report processed routes."
            )
        if complete != (not next_cursor):
            raise SpecValidationError(
                "Ecosystem completion disagrees with next_route_cursor."
            )
        if disabled_page and next_cursor and offset >= declared_count:
            raise SpecValidationError(
                "Disabled ecosystem route cursor did not advance."
            )
        if disabled_page and not next_cursor and declared_count - offset > route_limit:
            raise SpecValidationError(
                "Disabled ecosystem route pagination ended before route completion."
            )

        current_route_sha = page.get("route_sha256")
        current_query_sha = page.get("query_sha256")
        if not isinstance(current_route_sha, str) or not current_route_sha:
            raise SpecValidationError("Ecosystem route_sha256 is required.")
        if not isinstance(current_query_sha, str) or not current_query_sha:
            raise SpecValidationError("Ecosystem query_sha256 is required.")
        if first_page is None:
            first_page = deepcopy(page)
            route_sha256 = current_route_sha
            query_sha256 = current_query_sha
            route_count = declared_count
        elif (
            current_route_sha != route_sha256
            or current_query_sha != query_sha256
            or declared_count != route_count
        ):
            raise SpecValidationError("Ecosystem route pages changed their source.")

        if any(not isinstance(item, dict) for item in raw_pages):
            raise SpecValidationError("Ecosystem discovery pages must be objects.")
        if any(not isinstance(item, dict) for item in raw_errors):
            raise SpecValidationError("Ecosystem discovery errors must be objects.")
        pages.extend(deepcopy(raw_pages))
        errors.extend(deepcopy(raw_errors))
        statuses.append(status)
        candidate_count += declared_candidates
        processed_route_count += processed
        route_receipts.append(
            {
                "route_offset": offset,
                "advanced_to_offset": next_offset,
                "processed_route_count": processed,
                "candidate_count": declared_candidates,
            }
        )

        if planning_seed_only:
            planning_seed = deepcopy(first_page)
            planning_seed["aggregate_schema_version"] = "mmm/ecosystem-planning-seed-v1"
            planning_seed["status"] = _aggregate_ecosystem_status(
                statuses, pages, candidate_count
            )
            planning_seed["candidate_count"] = candidate_count
            planning_seed["pages"] = pages
            planning_seed["errors"] = errors
            planning_seed["coverage"] = (
                "planning seed only; remaining route catalog and provider cursors are "
                "intentionally deferred to specialist dependency/asset research"
            )
            planning_seed["collection_receipt"] = {
                "schema_version": "mmm/ecosystem-route-collection-receipt-v1",
                "route_page_count": 1,
                "route_limit": route_limit,
                "route_pages_sha256": _sha256(route_receipts),
                "planning_seed_only": True,
            }
            return planning_seed

        if not next_cursor:
            if not disabled_page and (
                remaining != 0 or next_offset != declared_count
            ):
                raise SpecValidationError(
                    "Ecosystem route pagination ended before route completion."
                )
            break
        if next_offset is not None and next_offset <= offset:
            raise SpecValidationError("Ecosystem route cursor did not advance.")
        if next_cursor == cursor or next_cursor in seen_cursors:
            raise SpecValidationError("Ecosystem route pagination repeated a cursor.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        previous_offset = offset
        expected_offset = next_offset

    assert first_page is not None and route_count is not None
    aggregate = first_page
    aggregate["aggregate_schema_version"] = "mmm/ecosystem-seed-aggregate-v1"
    aggregate["status"] = _aggregate_ecosystem_status(
        statuses, pages, candidate_count
    )
    aggregate["route_offset"] = 0
    aggregate["processed_route_count"] = processed_route_count
    aggregate["remaining_route_count"] = max(0, route_count - processed_route_count)
    aggregate["next_route_cursor"] = ""
    aggregate["routes_complete"] = True
    aggregate["candidate_count"] = candidate_count
    aggregate["pages"] = pages
    aggregate["errors"] = errors
    aggregate["collection_receipt"] = {
        "schema_version": "mmm/ecosystem-route-collection-receipt-v1",
        "route_page_count": len(route_receipts),
        "route_limit": route_limit,
        "route_pages_sha256": _sha256(route_receipts),
    }
    return aggregate


def _require_equal_page_invariants(
    first_page: Mapping[str, Any],
    page: Mapping[str, Any],
) -> None:
    """Reject cross-page drift outside pagination/results and receipt hashes."""
    volatile = {"requirements", "pagination", "radar_sha256"}
    first_keys = set(first_page) - volatile
    page_keys = set(page) - volatile
    if first_keys != page_keys:
        raise SpecValidationError("Technology pages changed invariant fields.")
    for key in sorted(first_keys):
        if first_page[key] != page[key]:
            raise SpecValidationError(
                f"Technology pages changed invariant field: {key}"
            )


def _strict_non_negative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SpecValidationError(f"{field} must be a non-negative integer.")
    return value


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _aggregate_ecosystem_status(
    statuses: list[str], pages: list[dict[str, Any]], candidate_count: int
) -> str:
    if candidate_count:
        return "available"
    if pages:
        return "empty"
    if statuses and all(status == "disabled" for status in statuses):
        return "disabled"
    return "unavailable"
