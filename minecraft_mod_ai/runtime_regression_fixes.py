from __future__ import annotations

"""Narrow current-runtime repairs for contracts that are still production-owned.

This module does not revive retired planner/bottleneck monkeypatch APIs. It only repairs
live public/runtime behavior that is exercised by the current production surface.
"""

import json
import math
import os
from dataclasses import replace
from functools import wraps
from typing import Any


_INSTALLED = False


def _fix_openai_planner() -> None:
    from . import planner as planner_module

    cls = planner_module.OpenAICompatiblePlanner
    if getattr(cls.__init__, "_mmm_timeout_bound_v1", False):
        return

    def init(
        self: Any,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        api_key = api_key.strip()
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_seconds must be a positive number.") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive number.")
        if not base_url.startswith("https://"):
            raise ValueError("외부 AI API 주소는 https://로 시작해야 합니다.")
        if not model:
            raise ValueError("외부 AI API 모델 이름을 입력해 주세요.")
        if not api_key:
            raise ValueError("외부 AI API 키가 없습니다.")
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout

    def plan(self: Any, prompt: str):
        request_body = planner_module.json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": planner_module._planner_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }
        ).encode("utf-8")
        request = planner_module.urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=request_body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "mmm-make-mincraft-mode/0.1",
            },
        )
        try:
            with planner_module.urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                payload_bytes = response.read(2 * 1024 * 1024 + 1)
        except (
            planner_module.urllib.error.HTTPError,
            planner_module.urllib.error.URLError,
        ) as exc:
            raise RuntimeError("외부 AI API 호출에 실패했습니다.") from exc
        if len(payload_bytes) > 2 * 1024 * 1024:
            raise RuntimeError("외부 AI API 응답이 너무 큽니다.")
        try:
            payload = planner_module.json.loads(payload_bytes.decode("utf-8"))
            content = payload["choices"][0]["message"]["content"]
        except (
            KeyError,
            IndexError,
            TypeError,
            UnicodeDecodeError,
            planner_module.json.JSONDecodeError,
        ) as exc:
            raise RuntimeError("외부 AI API 응답 형식이 올바르지 않습니다.") from exc
        if not isinstance(content, str):
            raise RuntimeError("외부 AI API가 텍스트 계획을 반환하지 않았습니다.")
        return planner_module._proposal_from_model_data(
            prompt, planner_module._extract_json_object(content)
        )

    init._mmm_timeout_bound_v1 = True
    plan._mmm_timeout_bound_v1 = True
    cls.__init__ = init
    cls.plan = plan


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

        search._mmm_openverse_route_v1 = True
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
                if not identifier or license_name not in {"cc0", "pdm", "by", "by-sa"}:
                    continue
                license_id = discovery._creative_commons_id(
                    license_name, license_version
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
                            item.get("detail_url"), allow_empty=True
                        ),
                        license_id=license_id,
                        license_url=discovery._safe_https_url(
                            item.get("license_url"), allow_empty=True
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
                            [item.get("thumbnail")], allowed_hosts=None
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

        search_openverse._mmm_openverse_candidates_v1 = True
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

    seed_query._mmm_lossless_seed_query_v1 = True
    discovery._seed_query = seed_query


def _fix_research_hotpaths() -> None:
    from . import centroid_vector_rag, rag_index, research_rag_performance
    from . import research_memory_performance, trajectory_memory

    if not getattr(
        research_rag_performance._ensure_semantic_lsh,
        "_mmm_no_blanket_delete_v1",
        False,
    ):
        perf = research_rag_performance

        def ensure_semantic_lsh(connection: Any) -> None:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mmm_semantic_lsh (
                    chunk_id TEXT PRIMARY KEY,
                    sig_a INTEGER NOT NULL,
                    sig_b INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS mmm_semantic_lsh_a "
                "ON mmm_semantic_lsh(sig_a)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS mmm_semantic_lsh_b "
                "ON mmm_semantic_lsh(sig_b)"
            )
            stale = connection.execute(
                """
                SELECT l.chunk_id
                FROM mmm_semantic_lsh AS l
                LEFT JOIN chunks AS c ON c.chunk_id = l.chunk_id
                WHERE c.chunk_id IS NULL
                LIMIT 2048
                """
            ).fetchall()
            if stale:
                connection.executemany(
                    "DELETE FROM mmm_semantic_lsh WHERE chunk_id = ?",
                    [(str(row[0]),) for row in stale],
                )

            batch_size = perf.env_int(
                "MMM_RAG_LSH_BUILD_BATCH", 256, minimum=32, maximum=2048
            )
            while True:
                rows = connection.execute(
                    """
                    SELECT c.chunk_id, c.embedding
                    FROM chunks AS c
                    LEFT JOIN mmm_semantic_lsh AS l ON l.chunk_id = c.chunk_id
                    WHERE l.chunk_id IS NULL AND c.embedding != '[]'
                    ORDER BY c.id
                    LIMIT ?
                    """,
                    (batch_size,),
                ).fetchall()
                if not rows:
                    break
                valid_ids: list[str] = []
                vectors: list[list[float]] = []
                for row in rows:
                    vector = perf._parse_embedding(row[1])
                    if vector:
                        valid_ids.append(str(row[0]))
                        vectors.append(vector)
                if valid_ids:
                    signatures = perf._signatures(vectors)
                    connection.executemany(
                        "INSERT OR REPLACE INTO mmm_semantic_lsh"
                        "(chunk_id, sig_a, sig_b) VALUES (?, ?, ?)",
                        [
                            (chunk_id, sig_a, sig_b)
                            for chunk_id, (sig_a, sig_b) in zip(
                                valid_ids, signatures, strict=True
                            )
                        ],
                    )
                valid_set = set(valid_ids)
                invalid_ids = [
                    str(row[0]) for row in rows if str(row[0]) not in valid_set
                ]
                if invalid_ids:
                    connection.executemany(
                        "INSERT OR REPLACE INTO mmm_semantic_lsh"
                        "(chunk_id, sig_a, sig_b) VALUES (?, -1, -1)",
                        [(chunk_id,) for chunk_id in invalid_ids],
                    )
                connection.commit()

        ensure_semantic_lsh._mmm_no_blanket_delete_v1 = True
        perf._ensure_semantic_lsh = ensure_semantic_lsh

    research_rag_performance.harden(rag_index, centroid_vector_rag)
    research_memory_performance.harden(trajectory_memory)


def _fix_llama_runtime_tuning() -> None:
    from . import llama_server_runtime_tuning as runtime

    if not getattr(runtime._explicit_parallel, "_mmm_parallel_cap_v1", False):

        def explicit_parallel() -> int | None:
            raw = os.environ.get("MMM_LLAMA_PARALLEL", "").strip()
            if not raw:
                return None
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError("MMM_LLAMA_PARALLEL must be an integer") from exc
            return max(1, min(8, value))

        explicit_parallel._mmm_parallel_cap_v1 = True
        runtime._explicit_parallel = explicit_parallel

    if not getattr(
        runtime._parallel_resource_feasible,
        "_mmm_parallel_fit_v1",
        False,
    ):

        def parallel_resource_feasible(
            slots: int,
            config: Any,
            model_path: str | None,
            resources: Any,
        ) -> bool:
            slots = max(1, int(slots))
            if slots == 1:
                return True
            context = runtime._per_request_context(config)
            try:
                total_context = runtime._total_context(context, slots)
            except RuntimeError:
                return False
            model_bytes = runtime._model_size(model_path) or 6 * 1024**3
            gpu_free = resources.gpu_free_bytes or (
                14 * 1024**3 if resources.gpu_total_bytes else 0
            )
            ram_avail = resources.ram_available_bytes or 12 * 1024**3
            if not gpu_free or not ram_avail:
                return False
            gpu_required = (
                int(model_bytes * 1.02)
                + total_context * runtime._kv_bytes_per_token()
                + 256 * 1024**2
            )
            ram_required = int(model_bytes * 0.30) + (
                512 + 256 * slots
            ) * 1024**2
            return bool(
                gpu_required <= int(gpu_free * 0.95)
                and ram_required <= int(ram_avail * 0.95)
            )

        parallel_resource_feasible._mmm_parallel_fit_v1 = True
        runtime._parallel_resource_feasible = parallel_resource_feasible

    current_install = runtime.install
    if getattr(current_install, "_mmm_launch_geometry_v1", False):
        return

    def fix_installed(autotune_module: Any) -> None:
        installed_launch = getattr(autotune_module, "_launch_selected", None)
        if installed_launch is None or getattr(
            installed_launch, "_mmm_launch_geometry_v1", False
        ):
            return
        base_launch = getattr(installed_launch, "__wrapped__", None)
        if not callable(base_launch):
            return

        @wraps(installed_launch)
        def launch_selected(
            binary: str,
            model_path: str,
            config: Any,
            selected: Any,
        ) -> str:
            explicit = runtime._explicit_parallel()
            requested = (
                explicit
                if explicit is not None
                else max(1, int(getattr(selected, "parallel", 1) or 1))
            )
            if explicit is not None:
                attempts = [requested]
            else:
                attempts = []
                slots = requested
                while True:
                    attempts.append(slots)
                    if slots <= 1:
                        break
                    slots = max(1, slots // 2)

            failures: list[str] = []
            active = selected
            url = ""
            for slots in attempts:
                root_name = str(getattr(selected, "name", "baseline")).split(
                    "|p", 1
                )[0]
                active = replace(
                    selected,
                    name=root_name if slots == 1 else f"{root_name}|p{slots}",
                    parallel=slots,
                )
                try:
                    url = base_launch(binary, model_path, config, active)
                    break
                except Exception as exc:
                    failures.append(f"p{slots}: {type(exc).__name__}: {exc}")

            if not url:
                message = (
                    "native llama-server failed every measured slot launch: "
                    + " | ".join(failures)
                )
                fresh_resources = runtime._runtime_resources()
                if runtime._recoverable_resource_failure(
                    failures,
                    slots=min(attempts),
                    config=config,
                    model_path=model_path,
                    resources=fresh_resources,
                ):
                    raise runtime.RecoverableResourceLaunchError(message)
                raise RuntimeError(message)

            resources = runtime._runtime_resources()
            slots = max(1, int(active.parallel))
            context_per_slot = runtime._per_request_context(config)
            context_total = runtime._total_context(context_per_slot, slots)
            active_ubatch = active.ubatch or min(
                autotune_module._env_int("MMM_LLAMA_BATCH", 2048),
                autotune_module._env_int("MMM_LLAMA_UBATCH", 512),
            )
            kv_k = os.environ.get(
                "MMM_LLAMA_ACTIVE_CACHE_TYPE_K",
                os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"),
            ).strip().lower()
            kv_v = os.environ.get(
                "MMM_LLAMA_ACTIVE_CACHE_TYPE_V",
                os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"),
            ).strip().lower()
            prompt_cache_enabled = not (
                runtime._is_qwen35_mtp_config(config)
                and os.environ.get("MMM_QWEN35_MTP_HOTPATH", "1")
                .strip()
                .lower()
                not in {"0", "false", "no", "off"}
            )
            receipt = {
                "schema_version": "mmm/llama-runtime-receipt-v1",
                "performance_mode": runtime._performance_mode(),
                "slots": slots,
                "context_per_slot": context_per_slot,
                "context_total": context_total,
                "ubatch": active_ubatch,
                "kv_k": kv_k,
                "kv_v": kv_v,
                "spec_type": active.spec_type,
                "draft_n_max": int(active.draft_n_max),
                "draft_p_min": float(active.draft_p_min),
                "cache_reuse": int(active.cache_reuse),
                "prompt_cache": prompt_cache_enabled,
                "cache_ram_mib": (
                    runtime._cache_ram_mib() if prompt_cache_enabled else 0
                ),
                "resource_bucket": runtime._resource_bucket(resources),
            }
            receipt["selection_inputs_sha256"] = runtime._json_fingerprint(
                runtime._selection_inputs(config)
            )
            receipt["selection_sha256"] = runtime._json_fingerprint(receipt)
            encoded = json.dumps(
                receipt, sort_keys=True, separators=(",", ":")
            )
            os.environ["MMM_LLAMA_RUNTIME_RECEIPT"] = encoded
            autotune_module._MMM_LLAMA_RUNTIME_RECEIPT = receipt
            os.environ["MMM_LLAMA_ACTIVE_PARALLEL"] = str(slots)
            os.environ["MMM_LLAMA_ACTIVE_UBATCH"] = str(active_ubatch)
            os.environ["MMM_LLAMA_ACTIVE_CACHE_REUSE"] = str(active.cache_reuse)
            os.environ["MMM_LLAMA_ACTIVE_SPEC_TYPE"] = active.spec_type
            return url

        launch_selected._mmm_launch_geometry_v1 = True
        autotune_module._launch_selected = launch_selected

    @wraps(current_install)
    def install(autotune_module: Any) -> None:
        current_install(autotune_module)
        fix_installed(autotune_module)

    install._mmm_launch_geometry_v1 = True
    runtime.install = install

    try:
        from . import llama_server_autotune
    except Exception:
        return
    fix_installed(llama_server_autotune)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _fix_openai_planner()
    _fix_ecosystem_discovery()
    _fix_research_hotpaths()
    _fix_llama_runtime_tuning()
    _INSTALLED = True


__all__ = ["install"]
