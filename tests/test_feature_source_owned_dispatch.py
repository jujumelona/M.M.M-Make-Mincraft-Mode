"""Exercise host-level source ownership dispatch without model inference."""
from __future__ import annotations

import json

from minecraft_mod_ai.construction_host_source import compose_host_java_source


def test_unimplemented_runtime_producer_reaches_direct_source_author(
    tmp_path, monkeypatch,
):
    from minecraft_mod_ai import construction_host_source as host
    import minecraft_mod_ai.host_fabric_callback_abi as callbacks
    import minecraft_mod_ai.first_pass_java_source as first_pass
    import minecraft_mod_ai.construction_file_author as verifier
    import minecraft_mod_ai.construction_implementation as implementation
    import minecraft_mod_ai.construction_coverage as coverage

    monkeypatch.setattr(implementation, "construction_implementation_fingerprint",
                        lambda: "source-owner-test-v1")
    monkeypatch.setattr(callbacks, "callback_interfaces", lambda *_args, **_kwargs: [
        {"event_id": "verified:tick", "parameters": ["example.World"]}
    ])
    monkeypatch.setattr(host, "_build_host_java_prompt",
                        lambda _router, _spec, _context, **_kwargs: (
                            {}, "Implement original feature",
                            [{"behavior_id": "test-1", "intent": "Implement original feature",
                              "runtime_state_reads": ["unbuilt_state"]}],
                            {"materialized_prerequisite_api": [],
                             "inspected_target_api": {},
                             "host_state_work_order_scope": {}}, [],
                        ))
    called = []

    def author(_router, spec, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        assert "unbuilt_state" in payload["runtime_state_bindings"]
        assert payload["available_fabric_callbacks"][0]["event_id"] == "verified:tick"
        assert kwargs["construction_context"]["host_direct_feature_source"] is True
        assert kwargs["behaviors"][0]["behavior_id"] == "test-1"
        called.append(True)
        return "package example; public class Bridge {}"

    monkeypatch.setattr(first_pass, "compose_first_pass_java", author)
    monkeypatch.setattr(verifier, "_verify_full_source", lambda *_a, **_k: None)
    monkeypatch.setattr(coverage, "record_authored_source_calls", lambda *_a, **_k: None)

    spec = {
        "target_path": "src/main/java/example/Bridge.java",
        "requirement": "Implement original feature",
        "bindings": {
            "request_feature_entrypoint": True,
            "runtime_state_bindings_json": json.dumps({
                "unbuilt_state": {"binding": "world_state",
                                  "semantic_status": "host_classified_unbound_read"}}),
        },
    }
    context = {"target": {"loader": "fabric", "java_version": 25},
               "java_classpath": ["/fake"], "project_root": str(tmp_path),
               "symbols": {}, "prerequisite_hashes": {}}
    output = compose_host_java_source(
        object(), spec, context, package="example", name="Bridge",
        owner="example.Bridge",
        behaviors=[{"behavior_id": "test-1", "intent": "Implement original feature",
                    "runtime_state_reads": ["unbuilt_state"]}],
    )
    assert output == "package example; public class Bridge {}"
    assert called == [True]
