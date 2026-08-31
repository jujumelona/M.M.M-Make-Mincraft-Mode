from __future__ import annotations

from minecraft_mod_ai.pre_design_rag_fusion import fuse_grounded_domain_evidence


def test_zero_relevance_provider_noise_is_removed_before_page_fusion():
    grounded = {
        "queries": [
            {
                "query": "modular spaceship colony alien combat",
                "evidence_records": [
                    {
                        "source_type": "github",
                        "source_id": "github:noise/stockmarket",
                        "title": "Stock Market Bot",
                        "url": "https://github.com/noise/stockmarket",
                        "content": "Python finance dashboard with equities and portfolio charts.",
                    },
                    {
                        "source_type": "github",
                        "source_id": "github:relevant/space-mod",
                        "title": "Modular Spaceship Systems",
                        "url": "https://github.com/relevant/space-mod",
                        "content": "A modular spaceship supports colony travel and alien combat systems.",
                    },
                ],
                "github_provider_status": "available",
            }
        ]
    }

    fused = fuse_grounded_domain_evidence({}, grounded)
    records = fused["queries"][0]["evidence_records"]

    assert len(records) == 1
    assert records[0]["source_id"] == "github:relevant/space-mod"
    assert fused["fusion"]["zero_relevance_dropped_record_count"] == 1
    assert fused["retrieval_trace"][0]["zero_relevance_dropped"] == 1
