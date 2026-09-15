from minecraft_mod_ai.verifier_failure_policy import (
    Capability,
    FailureKind,
    PhaseSpec,
    RECOVER_PHASE,
    check_java_target,
    classify_verifier_failure,
    is_no_progress,
    make_progress_fingerprint,
    source_repair_allowed,
)


def test_release_not_found_is_toolchain_environment() -> None:
    message = "release 25 is not found in the system"
    assert classify_verifier_failure(message) is FailureKind.TOOLCHAIN_ENVIRONMENT
    assert not source_repair_allowed(classify_verifier_failure(message))


def test_true_source_diagnostic_is_source_defect() -> None:
    message = "Foo.java:12: error: cannot find symbol"
    assert classify_verifier_failure(message) is FailureKind.SOURCE_DEFECT
    assert source_repair_allowed(classify_verifier_failure(message))


def test_unknown_never_defaults_to_source_defect() -> None:
    assert classify_verifier_failure("unclassified verifier failure") is FailureKind.UNKNOWN
    assert not source_repair_allowed(FailureKind.UNKNOWN)


def test_recover_contract_always_has_read_and_edit_tools() -> None:
    assert Capability.EDIT_EXISTING_SOURCE in RECOVER_PHASE.capabilities
    assert "read_file" in RECOVER_PHASE.tools
    assert "apply_source_edit" in RECOVER_PHASE.tools


def test_invalid_edit_phase_contract_fails_before_model_call() -> None:
    phase = PhaseSpec(
        name="RECOVER",
        capabilities=frozenset({Capability.EDIT_EXISTING_SOURCE}),
        tools=frozenset({"verify"}),
    )
    try:
        phase.validate()
    except RuntimeError as exc:
        assert "apply_source_edit" in str(exc)
        assert "read_file" in str(exc)
    else:
        raise AssertionError("invalid phase/tool contract must fail immediately")


def test_java_preflight_fails_as_environment_before_generation() -> None:
    result = check_java_target(25, [17, 21])
    assert not result.ok
    assert result.failure_kind is FailureKind.TOOLCHAIN_ENVIRONMENT
    assert "25" in (result.diagnostic or "")


def test_java_preflight_passes_when_target_is_supported() -> None:
    result = check_java_target(21, [17, 21, 25])
    assert result.ok
    assert result.failure_kind is None


def test_semantic_no_progress_requires_identical_nontransient_state() -> None:
    previous = make_progress_fingerprint(
        failure_kind=FailureKind.SOURCE_DEFECT,
        diagnostic_code="cannot-find-symbol",
        phase="RECOVER",
        source_state="same source",
        artifact_state="same artifact",
        toolchain_state="jdk-21",
    )
    current = make_progress_fingerprint(
        failure_kind=FailureKind.SOURCE_DEFECT,
        diagnostic_code="cannot-find-symbol",
        phase="RECOVER",
        source_state="same source",
        artifact_state="same artifact",
        toolchain_state="jdk-21",
    )
    assert is_no_progress(previous, current, material_change=False)


def test_toolchain_change_is_progress_not_loop() -> None:
    previous = make_progress_fingerprint(
        failure_kind=FailureKind.TOOLCHAIN_ENVIRONMENT,
        diagnostic_code="release-not-found",
        phase="VERIFY",
        source_state="same source",
        artifact_state="same artifact",
        toolchain_state="jdk-21",
    )
    current = make_progress_fingerprint(
        failure_kind=FailureKind.TOOLCHAIN_ENVIRONMENT,
        diagnostic_code="release-not-found",
        phase="VERIFY",
        source_state="same source",
        artifact_state="same artifact",
        toolchain_state="jdk-25",
    )
    assert not is_no_progress(previous, current, material_change=False)


def test_transient_failure_is_not_semantic_no_progress() -> None:
    fingerprint = make_progress_fingerprint(
        failure_kind=FailureKind.TRANSIENT_INFRASTRUCTURE,
        diagnostic_code="timeout",
        phase="VERIFY",
        source_state="same source",
        artifact_state="same artifact",
        toolchain_state="same toolchain",
    )
    assert not is_no_progress(fingerprint, fingerprint, material_change=False)
