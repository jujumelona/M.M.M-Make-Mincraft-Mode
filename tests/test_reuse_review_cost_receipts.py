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


def test_unrelated_first_five_allow_only_one_extra_semantic_window(monkeypatch):
    calls = {"model": 0, "pages": []}
    monkeypatch.setattr(review, "_github_archive",
                        lambda _c, _o, _r: ("b" * 40, b"source"))
    def inventory(_capability, _archive, search_terms=(), evidence_offset=0):
        calls["pages"].append(evidence_offset)
        chosen = "Teleport" if evidence_offset else "Generic"
        return {"status": "inspected", "license_spdx": "MIT", "java_file_count": 120,
                "evidence": [{"path": f"src/main/java/{chosen}.java",
                              "symbols": [chosen], "file_sha256": "sha256:" + "a"*64,
                              "rank": 1,
                              "snippet": f"class {chosen} {{}}"}]}
    def judge(_r, _capability, evidence, _license):
        calls["model"] += 1
        selected = evidence[0]
        intent = "reuse" if "Teleport" in selected["path"] else "reject"
        return {"status": "semantic_reviewed", "intent": intent,
                "scores": {"functional_fit": 4},
                "selected_file": selected["path"],
                "selected_symbol": selected["symbols"][0]}
    monkeypatch.setattr(review, "inventory_from_archive", inventory)
    monkeypatch.setattr(review, "_judge_with_model", judge)
    rows = review.assess_ranked_sources(
        "teleport", [{"source_url": "https://github.com/a/large",
                      "source_id": "curseforge:123"}], router=object(), cache={},
    )
    assert calls == {"model": 2, "pages": [0, 5]}
    assert rows[0]["semantic"]["selected_symbol"] == "Teleport"
    assert rows[0]["model_judgement_calls"] == 2


def test_korean_capability_passes_curated_curseforge_english_hint(monkeypatch):
    seen = []
    monkeypatch.setattr(review, "_github_archive",
                        lambda *_a: ("c" * 40, b"source"))
    def inventory(feature, _archive, search_terms=()):
        seen.append((feature, tuple(search_terms)))
        return {"status": "source_unavailable", "license_spdx": "MIT",
                "evidence": [], "java_file_count": 0}
    monkeypatch.setattr(review, "inventory_from_archive", inventory)
    review.assess_ranked_sources(
        "순간 이동 포탈",
        [{"source_url": "https://github.com/team/teleport",
          "source_id": "curseforge:32",
          "title": "Teleport Portal",
          "summary": "Teleport players between locations"}],
    )
    assert seen and seen[0][0] == "순간 이동 포탈"
    assert "Teleport Portal" in seen[0][1]
    assert "Teleport players between locations" in seen[0][1]
