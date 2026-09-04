from __future__ import annotations

from pathlib import Path

PATH = Path("minecraft_mod_ai/pre_design_grounded_rag.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new)


def edit_function(
    text: str,
    name: str,
    edits: tuple[tuple[str, str, str], ...],
) -> str:
    start = text.index(f"def {name}(")
    next_def = text.find("\ndef ", start + 1)
    end = len(text) if next_def < 0 else next_def
    block = text[start:end]
    for old, new, label in edits:
        block = replace_once(block, old, new, f"{name}:{label}")
    return text[:start] + block + text[end:]


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''_MAX_SOURCE_WORKERS = max(
    1, min(16, int(os.environ.get("MMM_PREDESIGN_SOURCE_WORKERS", "8") or 8))
)
_UA =''',
        '''_MAX_SOURCE_WORKERS = max(
    1, min(16, int(os.environ.get("MMM_PREDESIGN_SOURCE_WORKERS", "8") or 8))
)
# Keep every authored search query, but bound expensive provider/detail fan-out.
# Search endpoints are relevance-ranked; exhaustive catalog paging duplicates evidence
# and multiplies the later single-slot model-read work without improving scope coverage.
_MAX_PROVIDER_RESULTS_PER_QUERY = max(
    1,
    min(
        24,
        int(os.environ.get("MMM_PREDESIGN_PROVIDER_RESULTS_PER_QUERY", "6") or 6),
    ),
)
_MAX_PROVIDER_SEARCH_PAGES = max(
    1,
    min(
        4,
        int(os.environ.get("MMM_PREDESIGN_PROVIDER_SEARCH_PAGES", "2") or 2),
    ),
)
_UA =''',
        "provider budget constants",
    )

    text = edit_function(
        text,
        "_search_modrinth",
        (
            (
                "    provider_total = 0\n    while True:\n",
                "    provider_total = 0\n"
                "    while (\n"
                "        search_requests < _MAX_PROVIDER_SEARCH_PAGES\n"
                "        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY\n"
                "    ):\n"
                "        page_size = min(100, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))\n",
                "bounded loop",
            ),
            ('                "limit": 100,\n', '                "limit": page_size,\n', "page size"),
            (
                "        if len(hits) < 100 and not provider_total:\n",
                "        if len(hits) < page_size and not provider_total:\n",
                "short page stop",
            ),
        ),
    )

    text = edit_function(
        text,
        "_search_curseforge",
        (
            (
                "    while True:\n        params = urllib.parse.urlencode(\n",
                "    while (\n"
                "        search_requests < _MAX_PROVIDER_SEARCH_PAGES\n"
                "        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY\n"
                "    ):\n"
                "        page_size = min(50, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))\n"
                "        params = urllib.parse.urlencode(\n",
                "bounded loop",
            ),
            ('                "pageSize": 50,\n', '                "pageSize": page_size,\n', "page size"),
            (
                "            if len(rows) < 50:\n",
                "            if len(rows) < page_size:\n",
                "short page stop",
            ),
        ),
    )

    text = edit_function(
        text,
        "_search_github",
        (
            (
                "    while True:\n        if disabled is not None and disabled():\n",
                "    while (\n"
                "        search_requests < _MAX_PROVIDER_SEARCH_PAGES\n"
                "        and len(records) < _MAX_PROVIDER_RESULTS_PER_QUERY\n"
                "    ):\n"
                "        if disabled is not None and disabled():\n",
                "bounded loop",
            ),
            (
                '        params = urllib.parse.urlencode(\n            {\n'
                '                "q": _query_terms(query) + " minecraft fabric mod",\n'
                '                "per_page": 100,\n',
                '        page_size = min(100, _MAX_PROVIDER_RESULTS_PER_QUERY - len(records))\n'
                '        params = urllib.parse.urlencode(\n            {\n'
                '                "q": _query_terms(query) + " minecraft fabric mod",\n'
                '                "per_page": page_size,\n',
                "page size",
            ),
        ),
    )

    text = edit_function(
        text,
        "_linked_github_sources",
        (
            (
                "        if repo and repo not in repositories:\n            repositories.append(repo)\n",
                "        if repo and repo not in repositories:\n"
                "            if len(repositories) >= _MAX_PROVIDER_RESULTS_PER_QUERY:\n"
                "                continue\n"
                "            repositories.append(repo)\n",
                "linked source budget",
            ),
        ),
    )

    text = replace_once(
        text,
        '''        "query_workers": min(_MAX_QUERY_WORKERS, len(unique_queries)) if unique_queries else 0,
        "external_source_count": external_count,
        "domains": out_domains,''',
        '''        "query_workers": min(_MAX_QUERY_WORKERS, len(unique_queries)) if unique_queries else 0,
        "provider_result_budget": _MAX_PROVIDER_RESULTS_PER_QUERY,
        "provider_search_page_budget": _MAX_PROVIDER_SEARCH_PAGES,
        "external_source_count": external_count,
        "domains": out_domains,''',
        "bundle budget receipt",
    )

    text = replace_once(
        text,
        '''    result: list[dict[str, Any]] = []
    for row in (
        grounded.get("queries", []) if isinstance(grounded.get("queries"), list) else []
    ):''',
        '''    result: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()
    for row in (
        grounded.get("queries", []) if isinstance(grounded.get("queries"), list) else []
    ):''',
        "cross-query body dedupe setup",
    )
    text = replace_once(
        text,
        '''            body = _body(record)
            if not body:
                continue
            result.append(
''',
        '''            body = _body(record)
            if not body:
                continue
            body_key = str(record.get("content_sha256") or "").strip() or _sha256_text(body)
            if body_key in seen_bodies:
                continue
            seen_bodies.add(body_key)
            result.append(
''',
        "cross-query body dedupe",
    )

    PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
