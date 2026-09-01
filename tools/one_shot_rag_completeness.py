from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise SystemExit(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


# 1) Pre-design local code RAG: pass the real router and expand only while coverage improves.
path = Path("minecraft_mod_ai/pre_design_grounded_rag.py")
text = path.read_text(encoding="utf-8")
if "from .model_router import ModelRouter\n" not in text:
    text = replace_once(
        text,
        "from .rag_index import ProjectRAGIndex\n",
        "from .model_router import ModelRouter\nfrom .rag_index import ProjectRAGIndex\n",
        label="ModelRouter import",
    )
start = text.index("def _search_code_index(")
end = text.index("\ndef _github_repo_from_url(", start)
replacement = '''def _search_code_index(
    index: Path | None,
    query: str,
    router: Any,
) -> dict[str, Any]:
    if index is None:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "not_indexed",
            "hits": [],
        }

    try:
        searcher = ProjectRAGIndex(index)
        model_router = router if isinstance(router, ModelRouter) else None
        semantic = model_router is not None
        rerank = model_router is not None
        limit = 8
        previous_ids: set[str] = set()
        attempts: list[dict[str, Any]] = []
        semantic_fallback = False
        result = None
        saturation_reason = "coverage_satisfied"

        while True:
            try:
                result = searcher.search_with_receipt(
                    query,
                    limit=limit,
                    router=model_router,
                    semantic=semantic,
                    rerank=rerank,
                )
            except ValueError as exc:
                if semantic and "no semantic embeddings" in str(exc).casefold():
                    semantic = False
                    semantic_fallback = True
                    continue
                raise

            hit_ids = {str(hit.chunk_id) for hit in result.hits}
            warnings = {str(item) for item in result.receipt.warnings}
            coverage_low = "coverage_below_route_threshold" in warnings
            attempts.append(
                {
                    "limit": limit,
                    "result_count": len(result.hits),
                    "coverage_score": float(result.receipt.coverage_score),
                    "coverage_low": coverage_low,
                }
            )
            if not coverage_low:
                saturation_reason = "coverage_satisfied"
                break
            if len(result.hits) < limit:
                saturation_reason = "relevant_result_space_exhausted"
                break
            if previous_ids and not (hit_ids - previous_ids):
                saturation_reason = "no_new_relevant_hits"
                break
            previous_ids = hit_ids
            limit *= 2

        assert result is not None
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "searched",
            "hits": [asdict(hit) for hit in result.hits],
            "receipt": asdict(result.receipt),
            "adaptive_attempts": attempts,
            "saturation_reason": saturation_reason,
            "semantic_fallback": semantic_fallback,
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/code-rag-query-v3",
            "status": "error",
            "hits": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

'''
text = text[:start] + replacement + text[end + 1 :]
text = replace_once(
    text,
    '"code_rag": _search_code_index(code_index, query),',
    '"code_rag": _search_code_index(code_index, query, router),',
    label="code RAG router call",
)
path.write_text(text, encoding="utf-8")


# 2) Targeted GitHub retrieval delegates to the canonical adaptive retriever.
path = Path("minecraft_mod_ai/research_grounded_rag_contract.py")
text = path.read_text(encoding="utf-8")
start = text.index("def _github_targeted_search(")
end = text.index("\ndef ", start + 10)
replacement = '''def _github_targeted_search(
    query: str, *, seed_repositories: Sequence[Any] = ()
) -> dict[str, Any]:
    """Resolve a concrete source gap through the canonical adaptive GitHub retriever."""

    return _github_adaptive_search(
        query,
        seed_repositories=seed_repositories,
        search_if_needed=not bool(seed_repositories),
    )

'''
text = text[:start] + replacement + text[end + 1 :]
path.write_text(text, encoding="utf-8")


# 3) Legacy project-RAG fallback must index every eligible file and every chunk.
path = Path("minecraft_mod_ai/rag_index.py")
text = path.read_text(encoding="utf-8")
start = text.index("    def _load_legacy(self) -> list[RAGChunk]:")
end = text.index("\n\ndef _initialize_sqlite(", start)
replacement = '''    def _load_legacy(self) -> list[RAGChunk]:
        default_meta = {
            "minecraft_version": "1.21.1",
            "loader": "fabric",
            "mapping_namespace": "yarn",
            "java_version": "21",
            "license": "project-local",
            "source_commit": "HEAD",
            "path": str(self.index_path),
        }

        def source_chunks(file_path: Path) -> list[RAGChunk]:
            chunks: list[RAGChunk] = []
            for start_line, end_line, chunk_text in _chunk_file(file_path):
                digest = hashlib.sha256(
                    (
                        str(file_path)
                        + "\\0"
                        + str(start_line)
                        + "\\0"
                        + chunk_text
                    ).encode("utf-8")
                ).hexdigest()
                chunks.append(
                    RAGChunk(
                        chunk_id=f"sha256:{digest}",
                        source_path=str(file_path),
                        text=chunk_text,
                        start_line=start_line,
                        end_line=end_line,
                        sha256="sha256:" + hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                        metadata={**default_meta, "path": str(file_path)},
                        embedding=(),
                    )
                )
            return chunks

        if self.index_path.is_dir():
            chunks: list[RAGChunk] = []
            for file_path in _walk_files(self.index_path):
                if (
                    file_path.is_file()
                    and not file_path.is_symlink()
                    and file_path.suffix.lower() in _ALLOWED_SUFFIXES
                ):
                    chunks.extend(source_chunks(file_path))
            return chunks

        content = self.index_path.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            return []
        try:
            raw = json.loads(content)
            if isinstance(raw, dict) and raw.get("schema_version") == _LEGACY_SCHEMA_VERSION:
                result: list[RAGChunk] = []
                for item in raw.get("chunks", []):
                    result.append(
                        RAGChunk(
                            chunk_id=item["chunk_id"],
                            source_path=item["source_path"],
                            text=item["text"],
                            start_line=int(item["start_line"]),
                            end_line=int(item["end_line"]),
                            sha256=item["sha256"],
                            metadata=dict(item["metadata"]),
                            embedding=tuple(float(value) for value in item.get("embedding", [])),
                        )
                    )
                return result
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return source_chunks(self.index_path)
'''
text = text[:start] + replacement + text[end:]
path.write_text(text, encoding="utf-8")


# 4) Focused regressions for adaptive code RAG and lossless legacy fallback.
path = Path("tests/test_predesign_code_rag_adaptive.py")
path.write_text('''from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import pre_design_grounded_rag as rag
from minecraft_mod_ai.model_router import ModelRouter
from minecraft_mod_ai.rag_index import RAGHit, RAGSearchReceipt, RAGSearchResult


def _hit(index: int) -> RAGHit:
    return RAGHit(
        chunk_id=f"chunk-{index}",
        source_path=f"src/File{index}.java",
        start_line=1,
        end_line=2,
        score=1.0,
        lexical_score=1.0,
        semantic_score=0.0,
        reranker_score=0.0,
        text=f"class File{index} {{}}",
        metadata={},
    )


def _receipt(*, count: int, coverage: float, low: bool, semantic: bool, rerank: bool) -> RAGSearchReceipt:
    return RAGSearchReceipt(
        schema_version="mmm/rag-search-receipt-v1",
        query="target behavior",
        route="single",
        corrected_query=None,
        correction_applied=False,
        lexical_backend="test",
        semantic_requested=semantic,
        semantic_used=semantic,
        rerank_requested=rerank,
        rerank_used=rerank,
        candidates_considered=count,
        relation_expansions=0,
        result_count=count,
        query_terms=("target", "behavior"),
        covered_terms=("target",) if low else ("target", "behavior"),
        missing_terms=("behavior",) if low else (),
        coverage_score=coverage,
        relevance_score=1.0,
        required_metadata={},
        warnings=("coverage_below_route_threshold",) if low else (),
    )


def test_predesign_code_rag_passes_router_and_expands_until_coverage(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeIndex:
        def __init__(self, _path):
            pass

        def search_with_receipt(self, _query, **kwargs):
            calls.append(dict(kwargs))
            limit = int(kwargs["limit"])
            if limit == 8:
                count, low, coverage = 8, True, 0.4
            else:
                count, low, coverage = 10, False, 0.8
            semantic = bool(kwargs["semantic"])
            rerank = bool(kwargs["rerank"])
            return RAGSearchResult(
                hits=tuple(_hit(i) for i in range(count)),
                receipt=_receipt(
                    count=count,
                    coverage=coverage,
                    low=low,
                    semantic=semantic,
                    rerank=rerank,
                ),
            )

    monkeypatch.setattr(rag, "ProjectRAGIndex", FakeIndex)
    router = object.__new__(ModelRouter)
    result = rag._search_code_index(tmp_path / "index.sqlite", "target behavior", router)

    assert result["status"] == "searched"
    assert [item["limit"] for item in result["adaptive_attempts"]] == [8, 16]
    assert result["saturation_reason"] == "coverage_satisfied"
    assert len(result["hits"]) == 10
    assert calls[0]["router"] is router
    assert calls[0]["semantic"] is True
    assert calls[0]["rerank"] is True


def test_predesign_code_rag_falls_back_to_lexical_when_index_has_no_embeddings(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    class FakeIndex:
        def __init__(self, _path):
            pass

        def search_with_receipt(self, _query, **kwargs):
            calls.append(dict(kwargs))
            if kwargs["semantic"]:
                raise ValueError("semantic=True was requested, but this index has no semantic embeddings.")
            return RAGSearchResult(
                hits=(_hit(0),),
                receipt=_receipt(count=1, coverage=1.0, low=False, semantic=False, rerank=True),
            )

    monkeypatch.setattr(rag, "ProjectRAGIndex", FakeIndex)
    router = object.__new__(ModelRouter)
    result = rag._search_code_index(tmp_path / "index.sqlite", "target behavior", router)

    assert result["status"] == "searched"
    assert result["semantic_fallback"] is True
    assert calls[-1]["semantic"] is False
    assert calls[-1]["rerank"] is True
    assert calls[-1]["router"] is router
''', encoding="utf-8")

path = Path("tests/test_rag_index.py")
text = path.read_text(encoding="utf-8")
if "test_legacy_directory_fallback_has_no_50_file_cap" not in text:
    text += '''


def test_legacy_directory_fallback_has_no_50_file_cap(tmp_path: Path) -> None:
    for index in range(55):
        (tmp_path / f"File{index:02d}.java").write_text(
            f"class File{index:02d} {{}}", encoding="utf-8"
        )
    chunks = ProjectRAGIndex(tmp_path)._load_legacy()
    paths = {Path(chunk.source_path).name for chunk in chunks}
    assert "File54.java" in paths
    assert len(paths) == 55


def test_legacy_direct_source_preserves_tail_beyond_8192_characters(tmp_path: Path) -> None:
    source = tmp_path / "Large.java"
    source.write_text(
        "class Large {\\n" + ("int filler = 0;\\n" * 900) + "void tailMarker() {}\\n}",
        encoding="utf-8",
    )
    chunks = ProjectRAGIndex(source)._load_legacy()
    assert any("tailMarker" in chunk.text for chunk in chunks)
'''
path.write_text(text, encoding="utf-8")
