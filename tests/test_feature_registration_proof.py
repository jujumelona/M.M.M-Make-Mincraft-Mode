"""A successfully compiling registrar must actually register a checked listener."""
import hashlib

import pytest

from minecraft_mod_ai.feature_registration_proof import validate_registered_event_sources


OWNER = "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents"
FIELD = "END_SERVER_TICK"


def _source(*, with_import=True, registration=True, handler=True):
    suffix = hashlib.sha256(b"generic").hexdigest()[:20]
    prefix = ("import " + OWNER + ";\n") if with_import else ""
    call = (f"ServerTickEvents.{FIELD}.register(this::handleBehavior{suffix});"
            if registration else f"this.handleBehavior{suffix}(null);")
    if not handler:
        call = f"ServerTickEvents.{FIELD}.register(s -> {{ }});"
    return (prefix + "public class C {\n"
            f"private void handleBehavior{suffix}(Object event) {{ }}\n"
            f"private void registerBehavior{suffix}() {{ {call} }}\n"
            f"public void onInitialize() {{ this.registerBehavior{suffix}(); }}\n"
            "}")


def _events():
    return [{"event_id": OWNER + "#" + FIELD,
             "event_owner": OWNER, "event_field": FIELD}]


def test_exact_inspected_listener_registration_is_connected():
    validate_registered_event_sources(_source(), [{"behavior_id": "generic"}], _events())


def test_direct_one_shot_handler_call_is_not_an_event():
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            _source(registration=False), [{"behavior_id": "generic"}], _events())


def test_unimported_name_does_not_prove_correct_listener_owner():
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            _source(with_import=False), [{"behavior_id": "generic"}], _events())


def test_registering_a_listener_which_never_calls_the_feature_is_not_coverage():
    with pytest.raises(ValueError, match="HANDLER_NOT_CONNECTED"):
        validate_registered_event_sources(
            _source(handler=False), [{"behavior_id": "generic"}], _events())


def test_nonexistent_abi_event_cannot_pass_even_if_source_uses_that_name():
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            _source(), [{"behavior_id": "generic"}],
            [{"event_id": "unrelated#EVENT", "event_owner": "unrelated", "event_field": "EVENT"}])


def test_comment_does_not_satisfy_registration():
    source = _source(registration=False)
    source = source.replace("this.handleBehavior",
                            "// ServerTickEvents.END_SERVER_TICK.register(this::handleBehaviorNOPE);\nthis.handleBehavior")
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            source, [{"behavior_id": "generic"}], _events())


def test_separate_direct_handler_call_cannot_mask_empty_registered_listener():
    """A registrar cannot satisfy coverage by calling its handler once on init."""
    suffix = hashlib.sha256(b"generic").hexdigest()[:20]
    src = _source(handler=False)
    src = src.replace(
        f"private void registerBehavior{suffix}() {{ ",
        f"private void registerBehavior{suffix}() {{ this.handleBehavior{suffix}(null); ",
    )
    with pytest.raises(ValueError, match="INSPECTED_HOOK_NOT_REGISTERED"):
        validate_registered_event_sources(
            src, [{"behavior_id": "generic"}], _events())


def test_verified_lambda_forwarding_to_exact_handler_is_supported():
    suffix = hashlib.sha256(b"generic").hexdigest()[:20]
    src = _source().replace(
        f"register(this::handleBehavior{suffix})",
        f"register(s -> this.handleBehavior{suffix}(s))",
    )
    validate_registered_event_sources(src, [{"behavior_id": "generic"}], _events())
