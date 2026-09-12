from copy import deepcopy
import json

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import design_record_runtime as runtime
from minecraft_mod_ai import single_record_template as single


IDENTIFIER = "feature/behavior_contract/entry_conditions"


def _record(index: int) -> dict[str, str]:
    return {"trigger": f"trigger-{index}", "owner": "server"}


def _generator_for(count: int, *, fail_record_index: int | None = None, calls=None):
    calls = calls if calls is not None else []

    def generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, kwargs
        context = json.loads(messages[-1]["content"])
        if set(response_schema.get("properties", {})) == {"count", "blocked_reason"}:
            calls.append(("count", None))
            return {"count": count, "blocked_reason": ""}
        index = int(context["record_index"])
        calls.append(("record", index))
        if fail_record_index is not None and index == fail_record_index:
            raise TimeoutError("transport interrupted")
        return _record(index)

    return generate


def _patch_generator(monkeypatch, generate):
    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)
    monkeypatch.setattr(single, "generate_fixed_template_value", generate)


def test_interruption_resumes_cardinality_and_completed_ordinals(monkeypatch):
    progress: dict = {}
    calls: list[tuple[str, int | None]] = []
    _patch_generator(monkeypatch, _generator_for(2, fail_record_index=1, calls=calls))
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )

    with pytest.raises(TimeoutError):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)
    assert calls == [("count", None), ("record", 0), ("record", 1)]
    assert any(key.startswith("record-cardinality-v1:") for key in progress)
    assert any(key.startswith("single:") for key in progress)

    resumed_calls: list[tuple[str, int | None]] = []
    _patch_generator(monkeypatch, _generator_for(2, calls=resumed_calls))
    result = runtime.run_record_template(None, IDENTIFIER, **kwargs)
    assert result["records"] == [_record(0), _record(1)]
    assert resumed_calls == [("record", 1)]

    _patch_generator(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("completed host-owned record concern regenerated"),
    )
    assert runtime.run_record_template(None, IDENTIFIER, **kwargs) == result


@pytest.mark.parametrize("changed", ["context", "allowed_refs"])
def test_changed_inputs_do_not_reuse_prior_cardinality_or_records(monkeypatch, changed):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator_for(1))
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)

    if changed == "context":
        kwargs["context"] = {"criterion": "B"}
    else:
        kwargs["allowed_refs"] = {"new_source"}

    def must_regenerate(*args, **kwargs):
        raise StopIteration("binding changed")

    _patch_generator(monkeypatch, must_regenerate)
    with pytest.raises(StopIteration, match="binding changed"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_changed_template_contract_invalidates_saved_progress(monkeypatch):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator_for(1))
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)

    original_runtime_loader = runtime.load_record_template
    original_bounded_loader = bounded.load_record_template
    original_single_loader = single.load_record_template

    def changed(loader):
        def load(identifier):
            template = loader(identifier)
            template["task"] += " Clarified instruction."
            return template
        return load

    monkeypatch.setattr(runtime, "load_record_template", changed(original_runtime_loader))
    monkeypatch.setattr(bounded, "load_record_template", changed(original_bounded_loader))
    monkeypatch.setattr(single, "load_record_template", changed(original_single_loader))
    _patch_generator(
        monkeypatch,
        lambda *args, **kwargs: (_ for _ in ()).throw(StopIteration("contract changed")),
    )
    with pytest.raises(StopIteration, match="contract changed"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_saved_record_is_revalidated_before_any_model_call(monkeypatch):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator_for(1))
    kwargs = dict(
        context={},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)
    record_key = next(key for key in progress if key.startswith("single:"))
    progress[record_key]["trigger"] = "   "
    _patch_generator(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("invalid saved record must fail before model call"),
    )
    with pytest.raises(ValueError, match="SINGLE_TEMPLATE_RECORD"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_host_cardinality_has_no_legacy_128_record_completion_limit(monkeypatch):
    calls: list[tuple[str, int | None]] = []
    _patch_generator(monkeypatch, _generator_for(129, calls=calls))
    result = runtime.run_record_template(None, IDENTIFIER, context={}, allowed_refs=set())
    assert len(result["records"]) == 129
    assert calls[0] == ("count", None)
    assert calls[-1] == ("record", 128)


def test_invalid_record_is_never_checkpointed(monkeypatch):
    saved: list[tuple[str, object]] = []

    def generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        if set(response_schema.get("properties", {})) == {"count", "blocked_reason"}:
            return {"count": 1, "blocked_reason": ""}
        return {"trigger": "   ", "owner": "server"}

    _patch_generator(monkeypatch, generate)
    with pytest.raises(ValueError, match="SINGLE_TEMPLATE_RECORD"):
        runtime.run_record_template(
            None,
            IDENTIFIER,
            context={},
            allowed_refs=set(),
            checkpoint=lambda *args: saved.append(args),
        )
    assert any(key.startswith("record-cardinality-v1:") for key, _ in saved)
    assert not any(key.startswith("single:") for key, _ in saved)
