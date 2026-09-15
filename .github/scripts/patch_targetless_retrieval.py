from pathlib import Path

path = Path("minecraft_mod_ai/retrieval.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    '"""Target-bound official retrieval without historical platform defaults.\n\nThe static corpus is deliberately target-neutral. Exact Minecraft version, loader,\nmappings and toolchain coordinates are admitted only through the executable platform\nprovider selected by the host. Query text may rank trusted records but cannot invent\nor select a platform target.\n"""',
    '"""Official retrieval over target-neutral primary sources.\n\nA resolved Minecraft target is an optional precision filter, not a prerequisite.\nTargetless or partially resolved research still retrieves generic official evidence;\nversion-specific generation remains bound later by the executable platform provider.\n"""',
    1,
)

start = text.index("    def retrieve(\n", text.index("class OfficialCorpusIndex:"))
end = text.index("\n\ndef retrieve_official_evidence(", start)
method = '''    def retrieve(
        self,
        query: str,
        *,
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
        limit: int = 6,
    ) -> RetrievalReceipt:
        query = query.strip()
        if not 2 <= len(query) <= 2_000:
            raise SpecValidationError("RAG query length must be between 2 and 2000.")
        if type(limit) is not int or not 1 <= limit <= 12:
            raise SpecValidationError("RAG result limit must be between 1 and 12.")

        version = str(minecraft_version or "").strip()
        loader_id = str(loader or "").strip().casefold()
        mapping_id = str(mappings or "").strip()
        adapter = None
        if version and loader_id and mapping_id:
            try:
                candidate = adapter_for_target(version, loader_id)
            except ValueError:
                candidate = None
            if candidate is not None and mapping_id == candidate.yarn_mappings:
                adapter = candidate

        target_version = adapter.minecraft_version if adapter is not None else ""
        target_loader = adapter.loader if adapter is not None else ""
        target_mappings = adapter.yarn_mappings if adapter is not None else ""

        family = _classify_query(query)
        canonical = _canonical_query(query, family)
        eligible = {
            document.document_id: document
            for document in self.documents
            if adapter is None or document.loader in {adapter.loader, "agnostic"}
        }
        query_terms = frozenset(_tokens(canonical))
        query_grams = _trigrams(canonical)
        graph_boost: dict[str, float] = {document_id: 0.0 for document_id in eligible}
        lexical: dict[str, float] = {}
        semantic: dict[str, float] = {}
        family_score: dict[str, float] = {}
        for document_id, document in eligible.items():
            searchable = " ".join((document.title, document.content, *document.topics))
            document_terms = frozenset(_tokens(searchable))
            lexical[document_id] = len(query_terms & document_terms) / max(1, len(query_terms))
            semantic[document_id] = _jaccard(query_grams, _trigrams(searchable))
            family_score[document_id] = 1.0 if family in document.families else 0.0
        lexical_order = sorted(
            eligible,
            key=lambda document_id: (-lexical[document_id], document_id),
        )
        for rank, document_id in enumerate(lexical_order[:5], start=1):
            graph_boost[document_id] += 1.0 / rank
            for related_id in eligible[document_id].related_ids:
                if related_id in graph_boost:
                    graph_boost[related_id] += 0.45 / rank

        score: dict[str, float] = {}
        channels: dict[str, tuple[str, ...]] = {}
        for document_id in eligible:
            score[document_id] = (
                0.42 * lexical[document_id]
                + 0.28 * semantic[document_id]
                + 0.20 * family_score[document_id]
                + 0.10 * min(1.0, graph_boost[document_id])
            )
            active: list[str] = []
            if lexical[document_id] > 0:
                active.append("lexical")
            if semantic[document_id] > 0:
                active.append("semantic")
            if family_score[document_id] > 0:
                active.append("family")
            if graph_boost[document_id] > 0:
                active.append("graph")
            channels[document_id] = tuple(active)
        ordered = sorted(
            eligible,
            key=lambda document_id: (-score[document_id], document_id),
        )[:limit]

        target_receipt = (
            {
                "minecraft_version": target_version,
                "loader": target_loader,
                "mappings": target_mappings,
            }
            if adapter is not None
            else None
        )
        hits: list[RetrievalHit] = []
        for rank, document_id in enumerate(ordered, start=1):
            document = eligible[document_id]
            evidence_seed = canonical_json(
                {
                    "query": canonical,
                    "document_id": document_id,
                    "content_sha256": document.content_sha256,
                    "rank": rank,
                    "snapshot": self.snapshot_hash,
                    "target": target_receipt,
                }
            ).encode("utf-8")
            hits.append(
                RetrievalHit(
                    evidence_id="sha256:" + hashlib.sha256(evidence_seed).hexdigest(),
                    document_id=document_id,
                    title=document.title,
                    url=document.url,
                    excerpt=document.content,
                    content_sha256=document.content_sha256,
                    revision=document.revision,
                    minecraft_versions=("*",),
                    score=round(score[document_id], 8),
                    channels=channels[document_id],
                )
            )

        family_hits = sum(family in eligible[hit.document_id].families for hit in hits)
        signal_hits = sum(bool(hit.channels) for hit in hits)
        coverage = min(
            1.0,
            0.6 * family_hits / max(1, min(2, len(hits)))
            + 0.4 * signal_hits / max(1, min(3, len(hits))),
        )
        quality = (
            "strong"
            if hits and signal_hits >= min(2, len(hits)) and (family_hits > 0 or family == "project")
            else "weak"
        )
        correction_required = quality != "strong"
        if correction_required and adapter is not None:
            corrections = (
                f"{family} official API for Minecraft {target_version} {target_loader}",
                f"{family} mapping symbols for {target_mappings}",
                f"{family} deterministic runtime validation",
            )
        elif correction_required:
            corrections = (
                f"{family} official API concepts",
                f"{family} compatibility and mapping constraints",
                f"{family} deterministic runtime validation",
            )
        else:
            corrections = ()
        query_hash = "sha256:" + hashlib.sha256(
            canonical_json(
                {
                    "query": query,
                    "canonical": canonical,
                    "family": family,
                    "minecraft_version": target_version,
                    "loader": target_loader,
                    "mappings": target_mappings,
                }
            ).encode("utf-8")
        ).hexdigest()
        return RetrievalReceipt(
            schema_version="minecraft-mod-ai/retrieval-receipt-v1",
            query=query,
            canonical_query=canonical,
            query_family=family,
            minecraft_version=target_version,
            loader=target_loader,
            mappings=target_mappings,
            query_hash=query_hash,
            corpus_snapshot_hash=self.snapshot_hash,
            quality=quality,
            coverage=round(coverage, 6),
            correction_required=correction_required,
            correction_queries=corrections,
            hits=tuple(hits),
        )
'''
text = text[:start] + method + text[end:]

pub_start = text.index("def retrieve_official_evidence(")
pub_end = text.index("\n\ndef corpus_manifest()", pub_start)
public = '''def retrieve_official_evidence(
    query: str,
    *,
    minecraft_version: str | None = None,
    loader: str | None = None,
    mappings: str | None = None,
    limit: int = 6,
) -> RetrievalReceipt:
    with OfficialCorpusIndex() as index:
        return index.retrieve(
            query,
            minecraft_version=minecraft_version,
            loader=loader,
            mappings=mappings,
            limit=limit,
        )
'''
text = text[:pub_start] + public + text[pub_end:]
text = text.replace(
    '"target_policy": "explicit-provider-bound-only",',
    '"target_policy": "optional-refinement; targetless retrieval allowed",',
    1,
)
path.write_text(text, encoding="utf-8")
