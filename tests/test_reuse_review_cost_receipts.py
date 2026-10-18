"""Source inspection instrumentation: distinguish cache hits from model/API calls."""
from minecraft_mod_ai import curseforge_semantic_source_review as review


def test_source_review_cache_deduplicates_api_and_model(monkeypatch):
    calls = {"archive": 0, "model": 0}

    def archive(_client, _owner, _repository):
        calls["archive"] += 1
        return ("a" * 40, b"archive-bytes")

    def inventory(_feature, _archive, search_terms=()):
        return {
            "status": "inspected", "license_spdx": "MIT",
            "java_file_count": 1,
            "evidence": [{"path": "src/main/java/Teleport.java",
                          "symbols": ["Teleport"], "file_sha256": "sha256:" + "f"*64,
                          "snippet": "public class Teleport {}"}],
        }

    def judgement(_router, _feature, _evidence, _license):
        calls["model"] += 1
        return {"status": "semantic_reviewed",
                "selected_file": "src/main/java/Teleport.java", "selected_symbol": "Teleport"}

    monkeypatch.setattr(review, "_github_archive", archive)
    monkeypatch.setattr(review, "inventory_from_archive", inventory)
    monkeypatch.setattr(review, "_judge_with_model", judgement)

    candidate = [{"source_url": "https://github.com/official/teleport",
                  "source_id": "curseforge:11", "matched_queries": ["teleport"]}]
    cache = {}
    first = review.assess_ranked_sources("teleport", candidate, cache=cache, router=object())[0]
    again = review.assess_ranked_sources("teleport", candidate, cache=cache, router=object())[0]
    assert calls == {"archive": 1, "model": 1}
    assert first["source_fetches"] == first["model_judgement_calls"] == 1
    assert first["elapsed_ms"] >= 0
    assert again["source_cache_hits"] == again["semantic_cache_hits"] == 1
    assert again["source_fetches"] == again["model_judgement_calls"] == 0
    new_intent = review.assess_ranked_sources(
        "teleport with cooldown", candidate, cache=cache, router=object(),
    )[0]
    assert calls == {"archive": 1, "model": 2}
    assert new_intent["source_cache_hits"] == 1
    assert new_intent["semantic_cache_hits"] == 0
