from __future__ import annotations

from minecraft_mod_ai.procedural_retrieval import (
    decompose_task_procedure,
    extract_code_procedure,
    procedure_similarity,
    procedural_region_score,
)


def test_task_decomposition_is_ordered_and_domain_specific() -> None:
    plan = decompose_task_procedure(
        '{"kind":"networking","config":{"persist_state":true,"register_packet":true}}'
    )
    joined = " | ".join(plan.steps)
    assert plan.steps[0] == "locate existing contract"
    assert "encode and decode payload" in joined
    assert "register networking handler" in joined
    assert "load existing state" in joined
    assert plan.steps[-1] == "verify observable contract"


def test_code_procedure_preserves_execution_order() -> None:
    steps = extract_code_procedure(
        """
void handle() {
    State loaded = loadState();
    Packet encoded = encodePayload(loaded);
    NetworkChannel.send(encoded);
    Registry.registerHandler();
    validateState(loaded);
}
"""
    )
    assert steps
    assert steps.index("load state") < steps.index("encode decode payload")
    assert steps.index("encode decode payload") < steps.index("synchronize payload")
    assert steps.index("synchronize payload") < steps.index("register contract")


def test_procedure_similarity_rewards_order_not_identifier_overlap() -> None:
    target = (
        "load existing state",
        "encode and decode payload",
        "register networking handler",
        "verify observable contract",
    )
    same_flow = extract_code_procedure(
        """
void alpha() {
    Snapshot x = deserializeState();
    Message y = encodeMessage(x);
    Channel.sendPayload(y);
    Hooks.registerReceiver();
    checkResult(x);
}
"""
    )
    reversed_flow = extract_code_procedure(
        """
void beta() {
    checkResult(state);
    Hooks.registerReceiver();
    Channel.sendPayload(message);
    Snapshot state = deserializeState();
}
"""
    )
    assert procedure_similarity(target, same_flow) > procedure_similarity(target, reversed_flow)


def test_procedural_region_score_is_distinct_from_generic_semantic_similarity() -> None:
    plan = decompose_task_procedure("implement network packet sync and register handler")
    aligned, observed = procedural_region_score(
        plan,
        "void x(){ Packet p=encodePayload(state); channel.send(p); Registry.registerHandler(); validateState(state); }",
    )
    unrelated, _ = procedural_region_score(
        plan,
        "void x(){ int registryPacketCount = 1; String syncNetworkName = \"words only\"; }",
    )
    assert observed
    assert aligned > unrelated
    assert aligned > 0.0
