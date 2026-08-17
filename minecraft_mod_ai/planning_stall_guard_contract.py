from __future__ import annotations
'Keep planner research observable and prevent optional I/O from owning planning.\n\nThe canonical planner may use official/local evidence synchronously. Public ecosystem\nsearch and external MCP are optional evidence lanes: pre-design records their complete\nroute graph without performing network I/O, and specialist stages execute those routes\nonly when evidence is actually needed. Later explicit discovery keeps bounded provider\nI/O, connection reuse, and real parallel MCP sessions.\n'
import os
import threading
import time
import weakref
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any
_INSTALL_LOCK = threading.RLock()
_INSTALLED = False
_PROGRESS_WORKER_MARKER = '_mmm_planner_progress_worker_v1'
_PROGRESS_NORMALIZE_MARKER = '_mmm_planner_progress_normalize_v1'
_PROGRESS_MATERIALIZE_MARKER = '_mmm_planner_progress_materialize_v1'
_PROGRESS_PAGE_MARKER = '_mmm_planner_progress_page_v1'
_PROGRESS_WRITE_MARKER = '_mmm_planner_progress_write_v1'
_UNSET = object()
_EXTERNAL_PROVIDERS = frozenset({'modrinth', 'github', 'openverse_images', 'wikipedia', 'huggingface_models', 'openalex_works', 'crossref_works'})
_ALLOWED_API_HOSTS = frozenset({'api.modrinth.com', 'api.github.com', 'api.openverse.org', 'en.wikipedia.org', 'ko.wikipedia.org', 'huggingface.co', 'api.openalex.org', 'api.crossref.org'})

def _safe_progress_value(value: Any, *, fallback: str='-') -> str:
    """Keep progress output compact and prevent evidence/prompt text from leaking."""
    raw = str(value or '').strip()
    if not raw:
        return fallback
    cleaned = ''.join((char if char.isalnum() or char in {'-', '_', '.', ':', '/'} else '_' for char in raw))
    return cleaned[:80] or fallback

@dataclass
class _PlanningProgress:
    """Thread-safe, request-local progress shared with copied worker contexts."""
    stage: str = 'initializing'
    domain: str = '-'
    page: int = 0
    page_total: int = 0
    attempt: int = 0
    checkpoint: str = 'none'
    total: int = 0
    updated_at: float = field(default_factory=time.monotonic)
    _completed_domains: set[str] = field(default_factory=set, repr=False)
    _gap_domains: set[str] = field(default_factory=set, repr=False)
    _domain_attempts: dict[str, int] = field(default_factory=dict, repr=False)
    _page_attempts: dict[tuple[str, int, int], int] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def record(self, *, stage: Any=_UNSET, domain: Any=_UNSET, page: Any=_UNSET, page_total: Any=_UNSET, attempt: Any=_UNSET, checkpoint: Any=_UNSET, total: Any=_UNSET, complete_domain: str | None=None, gap_domain: str | None=None) -> dict[str, Any]:
        with self._lock:
            if stage is not _UNSET:
                self.stage = _safe_progress_value(stage, fallback='unknown')
            if domain is not _UNSET:
                self.domain = _safe_progress_value(domain)
            if page is not _UNSET:
                self.page = max(0, int(page))
            if page_total is not _UNSET:
                self.page_total = max(0, int(page_total))
            if attempt is not _UNSET:
                self.attempt = max(0, int(attempt))
            if checkpoint is not _UNSET:
                self.checkpoint = _safe_progress_value(checkpoint, fallback='none')
            if total is not _UNSET:
                self.total = max(0, int(total))
            if complete_domain:
                safe_complete = _safe_progress_value(complete_domain)
                self._completed_domains.add(safe_complete)
                self._gap_domains.discard(safe_complete)
            if gap_domain:
                safe_gap = _safe_progress_value(gap_domain)
                if safe_gap not in self._completed_domains:
                    self._gap_domains.add(safe_gap)
            self.updated_at = time.monotonic()
            return self._snapshot_locked()

    def begin_domain(self, domain: str) -> dict[str, Any]:
        safe_domain = _safe_progress_value(domain)
        with self._lock:
            invocation = self._domain_attempts.get(safe_domain, 0) + 1
            self._domain_attempts[safe_domain] = invocation
        return self.record(stage='domain-research' if invocation == 1 else 'domain-retry', domain=safe_domain, page=0, page_total=0, attempt=invocation, checkpoint='looking-up-checkpoint')

    def begin_page(self, domain: str, *, page: int, page_total: int, continuation_offset: int=0) -> dict[str, Any]:
        safe_domain = _safe_progress_value(domain)
        safe_page = max(1, int(page))
        safe_offset = max(0, int(continuation_offset))
        with self._lock:
            key = (safe_domain, safe_page, safe_offset)
            page_attempt = self._page_attempts.get(key, 0) + 1
            self._page_attempts[key] = page_attempt
        if page_attempt > 1:
            stage = 'page-retry'
        elif safe_offset > 0:
            stage = 'page-continuation'
        else:
            stage = 'page-research'
        return self.record(stage=stage, domain=safe_domain, page=safe_page, page_total=max(safe_page, int(page_total)), attempt=page_attempt, checkpoint=f'page-offset-{safe_offset}')

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {'stage': self.stage, 'domain': self.domain, 'page': self.page, 'page_total': self.page_total, 'attempt': self.attempt, 'checkpoint': self.checkpoint, 'completed': len(self._completed_domains), 'gaps': len(self._gap_domains), 'terminal': len(self._completed_domains | self._gap_domains), 'total': self.total, 'updated_at': self.updated_at}
_ACTIVE_PROGRESS: ContextVar[_PlanningProgress | None] = ContextVar('mmm_active_planner_progress', default=None)
_ACTIVE_PROGRESS_CURSOR: ContextVar[dict[str, Any] | None] = ContextVar('mmm_active_planner_progress_cursor', default=None)

def _progress_fields(progress: _PlanningProgress, *, now: float | None=None) -> str:
    snapshot = progress.snapshot()
    page_total = snapshot['page_total'] or '?'
    total = snapshot['total'] or '?'
    fields = f"stage={snapshot['stage']} domain={snapshot['domain']} page={snapshot['page']}/{page_total} attempt={snapshot['attempt']} checkpoint={snapshot['checkpoint']} completed={snapshot['completed']} gaps={snapshot['gaps']} terminal={snapshot['terminal']} total={total}"
    if now is not None:
        fields += f" idle={max(0.0, now - snapshot['updated_at']):.1f}s"
    return fields

def _emit_progress(progress: _PlanningProgress) -> None:
    print('planner research: pre-design progress', _progress_fields(progress), flush=True)

def report_planner_research_progress(*, stage: str | None=None, domain: str | None=None, page: int | None=None, page_total: int | None=None, attempt: int | None=None, checkpoint: str | None=None, completed_domain: str | None=None, gap_domain: str | None=None, total: int | None=None, emit: bool=True) -> bool:
    """Update the active plan's observable state without exposing research content.

    The function intentionally becomes a no-op outside an observed pre-design request,
    so low-level research/checkpoint code can report progress without owning lifecycle.
    """
    progress = _ACTIVE_PROGRESS.get()
    if progress is None:
        return False
    kwargs: dict[str, Any] = {}
    if stage is not None:
        kwargs['stage'] = stage
    if domain is not None:
        kwargs['domain'] = domain
    if page is not None:
        kwargs['page'] = page
    if page_total is not None:
        kwargs['page_total'] = page_total
    if attempt is not None:
        kwargs['attempt'] = attempt
    if checkpoint is not None:
        kwargs['checkpoint'] = checkpoint
    if total is not None:
        kwargs['total'] = total
    progress.record(complete_domain=completed_domain, gap_domain=gap_domain, **kwargs)
    if emit:
        _emit_progress(progress)
    return True

def _progress_int(value: Any, *, minimum: int=0) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(minimum, min(1000000000, parsed))

def _research_progress_hook(payload: Any) -> None:
    """Map paged-research events into the active request without owning execution."""
    if not isinstance(payload, dict) or _ACTIVE_PROGRESS.get() is None:
        return
    event = _safe_progress_value(payload.get('event'), fallback='research-event')
    cursor = dict(_ACTIVE_PROGRESS_CURSOR.get() or {})
    raw_domain = payload.get('domain_id')
    if raw_domain is not None:
        cursor['domain'] = _safe_progress_value(raw_domain, fallback='unknown')
    page = _progress_int(payload.get('page_index'), minimum=1)
    page_total = _progress_int(payload.get('page_count'), minimum=1)
    attempt = _progress_int(payload.get('attempt'), minimum=1)
    if page is not None:
        cursor['page'] = page
    if page_total is not None:
        cursor['page_total'] = page_total
    _ACTIVE_PROGRESS_CURSOR.set(cursor)
    stages = {'model_attempt': 'model-attempt', 'bounded_json_repair': 'bounded-json-repair', 'page_checkpoint_hit': 'page-checkpoint-hit', 'page_adaptive_split': 'page-adaptive-split', 'synthesis_checkpoint_hit': 'synthesis-checkpoint-hit', 'domain_checkpoint_complete': 'domain-checkpoint-hit', 'domain_start': 'domain-research', 'page_start': 'page-research', 'page_ledgered': 'page-ledgered', 'domain_complete': 'domain-complete', 'domain_gap_receipt': 'domain-gap-receipt'}
    checkpoint = event
    if event == 'model_attempt':
        checkpoint = 'model-requested'
    elif event == 'bounded_json_repair':
        checkpoint = 'repairing-bounded-json'
    elif event == 'page_checkpoint_hit':
        offset = _progress_int(payload.get('offset'))
        checkpoint = f'page-offset-{offset or 0}-loaded'
    elif event == 'page_adaptive_split':
        start = _progress_int(payload.get('start_offset'))
        midpoint = _progress_int(payload.get('midpoint'))
        end = _progress_int(payload.get('end_offset'))
        checkpoint = f'split-{start or 0}-{midpoint or 0}-{end or 0}'
    elif event == 'synthesis_checkpoint_hit':
        level = _progress_int(payload.get('level'))
        group = _progress_int(payload.get('group_index'))
        checkpoint = f'synthesis-{level or 0}-{group or 0}-loaded'
    elif event == 'domain_checkpoint_complete':
        checkpoint = 'domain-checkpoint-loaded'
    elif event == 'domain_start':
        checkpoint = 'domain-started'
    elif event == 'page_start':
        checkpoint = 'page-started'
    elif event == 'page_ledgered':
        checkpoint = 'page-ledger-saved'
    elif event == 'domain_complete':
        checkpoint = 'domain-saved'
    elif event == 'domain_gap_receipt':
        checkpoint = 'domain-gap-saved'
    successful_terminal = event in {'domain_checkpoint_complete', 'domain_complete'}
    gap_terminal = event == 'domain_gap_receipt'
    report_planner_research_progress(stage=stages.get(event, event), domain=cursor.get('domain'), page=cursor.get('page'), page_total=cursor.get('page_total'), attempt=attempt, checkpoint=checkpoint, completed_domain=cursor.get('domain') if successful_terminal else None, gap_domain=cursor.get('domain') if gap_terminal else None, total=_progress_int(payload.get('total'), minimum=1), emit=False)

def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, '').strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, '').strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

def _estimated_external_routes(research_brief: dict[str, Any]) -> tuple[int, int]:
    total = 0
    provider_slots = 0
    raw_domains = research_brief.get('domains')
    if not isinstance(raw_domains, list):
        return (0, 0)
    for raw_domain in raw_domains:
        if not isinstance(raw_domain, dict):
            continue
        queries = raw_domain.get('queries')
        providers = raw_domain.get('providers')
        if not isinstance(queries, list) or not isinstance(providers, list):
            continue
        external_count = len({str(provider) for provider in providers if str(provider) in _EXTERNAL_PROVIDERS})
        if external_count <= 0:
            continue
        provider_slots += external_count
        total += external_count * len(queries)
    return (total, provider_slots)

def _planning_seed_brief(research_brief: dict[str, Any]) -> dict[str, Any]:
    """Return a lossless planning projection; execution breadth is deferred, not cut."""
    projected = deepcopy(research_brief)
    estimated_routes, _provider_slots = _estimated_external_routes(projected)
    route_budget = _env_int('MMM_ECOSYSTEM_SEED_ROUTE_BUDGET', 96, minimum=16, maximum=512)
    projected['_mmm_planning_seed_projection'] = {'schema_version': 'mmm/planning-seed-projection-v3', 'estimated_external_routes': estimated_routes, 'route_budget': route_budget, 'compacted': False, 'reason': 'lossless_route_graph_external_io_deferred', 'queries_per_domain': None, 'full_research_brief_retained': True, 'domains_and_providers_preserved': True, 'specialist_discovery_continues_full_brief': True}
    return projected

def _brief_identity(research_brief: dict[str, Any]) -> str:
    sha = research_brief.get('brief_sha256')
    if isinstance(sha, str) and sha:
        target = research_brief.get('_mmm_platform_target')
        if isinstance(target, dict):
            target_key = tuple(sorted(((str(key), repr(value)) for key, value in target.items())))
        else:
            target_key = ()
        return f'{sha}|{target_key!r}'
    return f'id:{id(research_brief)}'

def _ecosystem_key(prompt: str, game_design: dict[str, Any], research_brief: dict[str, Any], page_builder: Any) -> tuple[str, int, str, int]:
    identity_brief = research_brief
    if not isinstance(research_brief.get('_mmm_platform_target'), dict):
        selection = game_design.get('_platform_selection')
        if isinstance(selection, dict) and isinstance(selection.get('target'), dict):
            identity_brief = {**research_brief, '_mmm_platform_target': dict(selection['target'])}
    return (prompt, id(game_design), _brief_identity(identity_brief), id(page_builder))

def _heartbeat(label: str, stop: threading.Event, started: float, progress: _PlanningProgress | None=None) -> None:
    interval = _env_float('MMM_PLANNER_HEARTBEAT_SECONDS', 15.0, minimum=5.0, maximum=60.0)
    while not stop.wait(interval):
        now = time.monotonic()
        progress_text = f' {_progress_fields(progress, now=now)}' if progress else ''
        print(f'planner research: {label} still running{progress_text} elapsed={now - started:.1f}s', flush=True)

def _patch_pre_design_progress_sources(agentic_module: Any, pre_design_module: Any) -> None:
    """Attach request-local progress probes to dynamically resolved research helpers."""
    progress_setter = getattr(pre_design_module, 'set_research_progress_hook', None)
    if callable(progress_setter):
        progress_setter(_research_progress_hook)
    current_normalize = agentic_module.normalize_research_brief
    if current_normalize.__dict__.get(_PROGRESS_NORMALIZE_MARKER) is not current_normalize:

        @wraps(current_normalize)
        def normalize_observed(*args: Any, **kwargs: Any) -> Any:
            result = current_normalize(*args, **kwargs)
            progress = _ACTIVE_PROGRESS.get()
            if progress is not None and isinstance(result, dict):
                domains = result.get('domains')
                total = sum((isinstance(item, dict) for item in domains)) if isinstance(domains, list) else 0
                progress.record(stage='research-brief', total=total, checkpoint='brief-ready')
                _emit_progress(progress)
            return result
        setattr(normalize_observed, _PROGRESS_NORMALIZE_MARKER, normalize_observed)
        agentic_module.normalize_research_brief = normalize_observed
    current_worker = agentic_module._research_domain_with_agent
    if current_worker.__dict__.get(_PROGRESS_WORKER_MARKER) is not current_worker:

        @wraps(current_worker)
        def worker_observed(router: Any, *, prompt: str, domain: Any, deterministic: Any, trace_metadata: Any) -> Any:
            domain_id = (str(domain.get('domain_id', '')).strip() if isinstance(domain, dict) else '') or 'unknown'
            progress = _ACTIVE_PROGRESS.get()
            cursor_token = None
            if progress is not None:
                cursor_token = _ACTIVE_PROGRESS_CURSOR.set({'domain': _safe_progress_value(domain_id)})
                progress.begin_domain(domain_id)
                _emit_progress(progress)
            try:
                try:
                    result = current_worker(router, prompt=prompt, domain=domain, deterministic=deterministic, trace_metadata=trace_metadata)
                except BaseException:
                    if progress is not None:
                        progress.record(stage='domain-recovery', domain=domain_id, checkpoint='last-safe-checkpoint')
                        _emit_progress(progress)
                    raise
                if progress is not None:
                    progress.record(stage='domain-complete', domain=domain_id, checkpoint='domain-saved', complete_domain=domain_id)
                    _emit_progress(progress)
                return result
            finally:
                if cursor_token is not None:
                    _ACTIVE_PROGRESS_CURSOR.reset(cursor_token)
        setattr(worker_observed, _PROGRESS_WORKER_MARKER, worker_observed)
        worker_observed.__wrapped__ = current_worker
        agentic_module._research_domain_with_agent = worker_observed
    current_materialize = getattr(pre_design_module, '_materialize_domain_evidence_document', None)
    if callable(current_materialize) and current_materialize.__dict__.get(_PROGRESS_MATERIALIZE_MARKER) is not current_materialize:

        @wraps(current_materialize)
        def materialize_observed(domain_id: str, evidence: Any) -> Any:
            progress = _ACTIVE_PROGRESS.get()
            if progress is not None:
                progress.record(stage='evidence-snapshot', domain=domain_id, checkpoint='saving-evidence')
                _emit_progress(progress)
            result = current_materialize(domain_id, evidence)
            if progress is not None and isinstance(result, dict):
                progress.record(stage='evidence-snapshot', domain=domain_id, page=0, page_total=max(0, int(result.get('page_count', 0) or 0)), checkpoint='evidence-saved')
                _emit_progress(progress)
            return result
        setattr(materialize_observed, _PROGRESS_MATERIALIZE_MARKER, materialize_observed)
        pre_design_module._materialize_domain_evidence_document = materialize_observed
    current_page_messages = getattr(pre_design_module, '_research_page_messages', None)
    if callable(current_page_messages) and current_page_messages.__dict__.get(_PROGRESS_PAGE_MARKER) is not current_page_messages:

        @wraps(current_page_messages)
        def page_messages_observed(*args: Any, **kwargs: Any) -> Any:
            domain = kwargs.get('domain')
            page_value = kwargs.get('page')
            document = kwargs.get('document')
            domain_id = (str(domain.get('domain_id', '')).strip() if isinstance(domain, dict) else '') or 'unknown'
            if isinstance(page_value, dict):
                page_index = int(page_value.get('page_index', 0) or 0) + 1
                page_total = int(page_value.get('page_count', 0) or 0)
            else:
                page_index = 1
                page_total = 0
            if page_total <= 0 and isinstance(document, dict):
                page_total = int(document.get('page_count', 0) or 0)
            progress = _ACTIVE_PROGRESS.get()
            if progress is not None:
                cursor = dict(_ACTIVE_PROGRESS_CURSOR.get() or {})
                cursor.update({'domain': _safe_progress_value(domain_id), 'page': page_index, 'page_total': max(page_index, page_total)})
                _ACTIVE_PROGRESS_CURSOR.set(cursor)
                progress.begin_page(domain_id, page=page_index, page_total=max(page_index, page_total), continuation_offset=max(0, int(kwargs.get('continuation_offset', 0) or 0)))
                _emit_progress(progress)
            return current_page_messages(*args, **kwargs)
        setattr(page_messages_observed, _PROGRESS_PAGE_MARKER, page_messages_observed)
        pre_design_module._research_page_messages = page_messages_observed
    current_write = getattr(pre_design_module, '_atomic_write_text', None)
    if callable(current_write) and current_write.__dict__.get(_PROGRESS_WRITE_MARKER) is not current_write:

        @wraps(current_write)
        def write_observed(path: Any, content: str) -> Any:
            progress = _ACTIVE_PROGRESS.get()
            filename = Path(str(path)).name.lower()
            is_checkpoint = 'checkpoint' in filename
            if progress is not None and is_checkpoint:
                progress.record(checkpoint='saving-checkpoint')
                _emit_progress(progress)
            result = current_write(path, content)
            if progress is not None and is_checkpoint:
                progress.record(checkpoint='checkpoint-saved')
                _emit_progress(progress)
            return result
        setattr(write_observed, _PROGRESS_WRITE_MARKER, write_observed)
        pre_design_module._atomic_write_text = write_observed

def _patch_pre_design_external_seed(agentic_module: Any, central_module: Any) -> None:
    """Record the full external route graph without doing public network I/O."""
    current = agentic_module.collect_ecosystem_seed_bundle
    if getattr(current, '_mmm_pre_design_external_deferred', False):
        return

    @wraps(current)
    def deferred_seed(prompt: str, game_design: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        research_brief = kwargs.get('research_brief')
        planning_seed_only = bool(kwargs.get('planning_seed_only', False))
        if not planning_seed_only or not isinstance(research_brief, dict):
            return current(prompt, game_design, *args, **kwargs)
        routes = central_module.external_discovery_routes(research_brief)
        route_receipts = [{'domain_id': str(route.get('domain_id', '')), 'provider': str(route.get('provider', '')), 'target_profile': str(route.get('target_profile', '')), 'query_sha256': central_module._sha256(str(route.get('query', '')))} for route in routes]
        route_sha256 = central_module._sha256(central_module.canonical_json(route_receipts))
        print('planner research: ecosystem network deferred', f' routes={len(routes)}', flush=True)
        return {'schema_version': 'mmm/ecosystem-planning-deferred-v1', 'status': 'deferred', 'brief_sha256': str(research_brief.get('brief_sha256', '')), 'route_sha256': route_sha256, 'route_count': len(routes), 'processed_route_count': 0, 'remaining_route_count': len(routes), 'routes_complete': not routes, 'candidate_count': 0, 'pages': [], 'errors': [], 'route_receipts': route_receipts, 'coverage': 'Complete external route graph retained; provider I/O is intentionally deferred to adaptive specialist research outside the planning critical path.', 'authorization': 'none', 'download_performed': False, 'planning_critical_path': False}
    deferred_seed._mmm_pre_design_external_deferred = True
    agentic_module.collect_ecosystem_seed_bundle = deferred_seed

def _patch_pre_design_observability(agentic_module: Any) -> None:
    current = agentic_module.collect_pre_design_research
    if current.__dict__.get('_mmm_pre_design_heartbeat') is current:
        return

    @wraps(current)
    def observed(router: Any, prompt: str, *, trace_metadata: dict[str, Any] | None=None) -> dict[str, Any]:
        started = time.monotonic()
        stop = threading.Event()
        progress = _PlanningProgress()
        progress_token = _ACTIVE_PROGRESS.set(progress)
        cursor_token = _ACTIVE_PROGRESS_CURSOR.set({})
        heartbeat = threading.Thread(target=_heartbeat, args=('pre-design', stop, started, progress), daemon=True, name='mmm_pre_design_heartbeat')
        print('planner research: pre-design start', _progress_fields(progress), flush=True)
        heartbeat.start()
        try:
            result = current(router, prompt, trace_metadata=trace_metadata)
        except BaseException:
            progress.record(stage='failed', checkpoint='last-safe-checkpoint')
            _emit_progress(progress)
            raise
        finally:
            stop.set()
            _ACTIVE_PROGRESS_CURSOR.reset(cursor_token)
            _ACTIVE_PROGRESS.reset(progress_token)
        final_snapshot = progress.snapshot()
        gap_count = int(final_snapshot.get('gaps', 0) or 0)
        if gap_count:
            progress.record(stage='complete-with-gaps', checkpoint='research-saved-with-gaps')
            final_label = 'planner research: pre-design terminal with gaps'
        else:
            progress.record(stage='complete', checkpoint='research-saved')
            final_label = 'planner research: pre-design complete'
        print(final_label, _progress_fields(progress), f'elapsed={time.monotonic() - started:.1f}s', flush=True)
        return result
    observed._mmm_pre_design_heartbeat = observed
    observed.__wrapped__ = current
    agentic_module.collect_pre_design_research = observed

def _patch_external_mcp_parallel(external_module: Any) -> None:
    """Allow independent MCP sessions to overlap instead of holding one global lock."""
    cls = external_module.ExternalMCPRouter
    current = cls._call_provider
    if getattr(current, '_mmm_parallel_sessions', False):
        return

    @wraps(current)
    def call_provider_parallel(self: Any, server_name: str, entry: dict[str, Any], *, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        import asyncio
        import anyio

        async def run() -> dict[str, Any]:
            return await self._call_provider_async(server_name, entry, tool=tool, arguments=arguments)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(run)
        value: dict[str, Any] = {}
        error: list[BaseException] = []

        def worker() -> None:
            try:
                value['result'] = anyio.run(run)
            except BaseException as exc:
                error.append(exc)
        thread = threading.Thread(target=worker, daemon=True, name=f'mmm_external_mcp_{server_name}')
        thread.start()
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise external_module.ExternalMCPError(f'External MCP {server_name} exceeded the synchronous bridge timeout.')
        if error:
            raise external_module.ExternalMCPError(str(error[0])) from error[0]
        return value['result']
    call_provider_parallel._mmm_parallel_sessions = True
    cls._call_provider = call_provider_parallel

def _patch_discovery_http_pool(ecosystem_module: Any) -> None:
    """Reuse one httpx connection pool across every route handled by one client."""
    cls = ecosystem_module.EcosystemDiscoveryClient
    current_init = cls.__init__
    current_get_json = cls._get_json
    if getattr(current_get_json, '_mmm_connection_pool', False):
        return

    @wraps(current_init)
    def init_with_pool(self: Any, *args: Any, **kwargs: Any) -> None:
        if 'timeout_seconds' not in kwargs:
            kwargs['timeout_seconds'] = _env_float('MMM_ECOSYSTEM_PROVIDER_TIMEOUT_SECONDS', 8.0, minimum=2.0, maximum=30.0)
        current_init(self, *args, **kwargs)
        pooled = ecosystem_module.httpx.Client(timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport)
        self._mmm_http_client = pooled
        self._mmm_http_finalizer = weakref.finalize(self, pooled.close)
    init_with_pool._mmm_planner_timeout_default = True
    init_with_pool._mmm_connection_pool = True

    @wraps(current_get_json)
    def get_json_pooled(self: Any, url: str, *, params: dict[str, str] | None=None, provider: str='', include_next_url: bool=False) -> Any:
        parsed = ecosystem_module.urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in _ALLOWED_API_HOSTS:
            raise ecosystem_module.SpecValidationError('Discovery request escaped the API allowlist.')
        if parsed.hostname == 'huggingface.co' and (not (parsed.path == '/api/models' or parsed.path.startswith('/api/models/'))):
            raise ecosystem_module.SpecValidationError('Hugging Face discovery is restricted to metadata API paths.')
        if parsed.hostname == 'api.openalex.org' and (not (parsed.path == '/works' or parsed.path.startswith('/works/'))):
            raise ecosystem_module.SpecValidationError('OpenAlex discovery is restricted to works metadata paths.')
        if parsed.hostname == 'api.crossref.org' and (not (parsed.path == '/works' or parsed.path.startswith('/works/'))):
            raise ecosystem_module.SpecValidationError('Crossref discovery is restricted to works metadata paths.')
        headers = {'Accept': 'application/json', 'User-Agent': ecosystem_module._USER_AGENT}
        if provider == 'github':
            headers['X-GitHub-Api-Version'] = '2022-11-28'
            headers['Accept'] = 'application/vnd.github+json'
            if self.github_token:
                headers['Authorization'] = f'Bearer {self.github_token}'
        elif provider == 'openverse' and self.openverse_token:
            headers['Authorization'] = f'Bearer {self.openverse_token}'
        client = getattr(self, '_mmm_http_client', None)
        if client is None:
            client = ecosystem_module.httpx.Client(timeout=self.timeout_seconds, follow_redirects=False, transport=self.transport)
            self._mmm_http_client = client
            self._mmm_http_finalizer = weakref.finalize(self, client.close)
        try:
            response = client.get(url, params=params, headers=headers)
        except ecosystem_module.httpx.HTTPError as exc:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery request failed: {type(exc).__name__}.') from exc
        if response.status_code != 200:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery returned HTTP {response.status_code}.')
        if len(response.content) > ecosystem_module._MAX_RESPONSE_BYTES:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery response exceeded the byte policy.')
        try:
            payload = response.json()
        except ValueError as exc:
            raise ecosystem_module.EcosystemDiscoveryUnavailable(f'{parsed.hostname} discovery returned invalid JSON.') from exc
        if include_next_url:
            next_link = response.links.get('next')
            next_url = str(next_link.get('url') or '') if isinstance(next_link, dict) else ''
            return (payload, next_url)
        return payload
    get_json_pooled._mmm_connection_pool = True
    cls.__init__ = init_with_pool
    cls._get_json = get_json_pooled

def _patch_worker_defaults(parallel_module: Any, agentic_module: Any) -> None:
    current_parallel = parallel_module._env_workers
    if not getattr(current_parallel, '_mmm_planning_io_tuned', False):

        @wraps(current_parallel)
        def tuned_parallel_workers(name: str, default: int, *, maximum: int=32) -> int:
            if not os.environ.get(name, '').strip():
                if name == 'MMM_DISCOVERY_WORKERS':
                    default = 24
                elif name == 'MMM_RESEARCH_WORKERS':
                    default = 16
            return current_parallel(name, default, maximum=maximum)
        tuned_parallel_workers._mmm_planning_io_tuned = True
        parallel_module._env_workers = tuned_parallel_workers
    current_agentic = agentic_module._env_workers
    if not getattr(current_agentic, '_mmm_planning_io_tuned', False):

        @wraps(current_agentic)
        def tuned_agentic_workers(name: str='MMM_RESEARCH_WORKERS', default: int=8) -> int:
            if name == 'MMM_RESEARCH_WORKERS' and (not os.environ.get(name, '').strip()):
                default = 16
            return current_agentic(name, default)
        tuned_agentic_workers._mmm_planning_io_tuned = True
        agentic_module._env_workers = tuned_agentic_workers

def _patch_complete_planner(complete_planner_module: Any) -> None:
    """Add observability only; never create another research future here."""
    current_impl = complete_planner_module._retrieve_implementation_evidence
    current_collect = complete_planner_module.collect_ecosystem_seed_bundle
    if getattr(current_impl, '_mmm_stall_guard', False):
        return

    @wraps(current_impl)
    def implementation_evidence_observed(prompt: str, game_design: dict[str, Any], research_brief: dict[str, Any] | None=None) -> dict[str, Any]:
        started = time.monotonic()
        result = current_impl(prompt, game_design, research_brief)
        print('planner research: official RAG complete', f' elapsed={time.monotonic() - started:.1f}s', sep='', flush=True)
        return result
    implementation_evidence_observed._mmm_stall_guard = True
    implementation_evidence_observed._mmm_parallel_target_rag = True
    implementation_evidence_observed._mmm_agentic_rag_fusion = True

    @wraps(current_collect)
    def ecosystem_seed_observed(prompt: str, game_design: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
        started = time.monotonic()
        stop = threading.Event()
        heartbeat = threading.Thread(target=_heartbeat, args=('ecosystem seed', stop, started), daemon=True, name='mmm_ecosystem_heartbeat')
        print('planner research: ecosystem seed join', flush=True)
        heartbeat.start()
        try:
            result = current_collect(prompt, game_design, *args, **kwargs)
        finally:
            stop.set()
        print('planner research: ecosystem seed complete', f" routes={result.get('route_count', 'unknown')}", f" processed={result.get('processed_route_count', 0)}", f" remaining={result.get('remaining_route_count', 0)}", f" candidates={result.get('candidate_count', 0)}", f' elapsed={time.monotonic() - started:.1f}s', sep='', flush=True)
        return result
    ecosystem_seed_observed._mmm_stall_guard = True
    ecosystem_seed_observed._mmm_parallel_planner_overlap = True
    complete_planner_module._retrieve_implementation_evidence = implementation_evidence_observed
    complete_planner_module.collect_ecosystem_seed_bundle = ecosystem_seed_observed

def install() -> None:
    """Install single-owner planner research I/O policy after canonical composition."""
    global _INSTALLED
    if _INSTALLED:
        return
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        from . import agentic_pre_design_rag, agentic_research_fusion, agentic_research_game_design, central_research, complete_planner, ecosystem_discovery, external_mcp_router, parallel_runtime_contract
        _patch_pre_design_external_seed(agentic_research_game_design, central_research)
        _patch_pre_design_progress_sources(agentic_research_game_design, agentic_pre_design_rag)
        _patch_pre_design_observability(agentic_research_game_design)
        _patch_external_mcp_parallel(external_mcp_router)
        _patch_discovery_http_pool(ecosystem_discovery)
        _patch_worker_defaults(parallel_runtime_contract, agentic_research_fusion)
        _patch_complete_planner(complete_planner)
        _INSTALLED = True
__all__ = ['_ecosystem_key', '_planning_seed_brief', 'install', 'report_planner_research_progress']
