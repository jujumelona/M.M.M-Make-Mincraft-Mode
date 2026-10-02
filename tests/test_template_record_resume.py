from copy import deepcopy

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import design_record_runtime as runtime


IDENTIFIER = "feature/behavior_contract/entry_conditions"


def _record(index: int) -> dict[str, str]:
    return {"trigger": f"trigger-{index}", "owner": "server"}


def _generator(records, *, fail=False, calls=None):
    calls = calls if calls is not None else []

    def generate(router, role, messages, *, response_schema, **kwargs):
        del router, role, messages, kwargs
        calls.append(response_schema)
        if fail:
            raise TimeoutError("transport interrupted")
        return {"records": deepcopy(records)}

    return generate


def _patch_generator(monkeypatch, generate):
    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)


def test_interruption_retries_one_atomic_record_set_without_partial_protocol_state(monkeypatch):
    progress: dict = {}
    failed_calls = []
    _patch_generator(monkeypatch, _generator([_record(0), _record(1)], fail=True, calls=failed_calls))
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )

    with pytest.raises(TimeoutError):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)
    assert len(failed_calls) == 1
    assert not progress

    resumed_calls = []
    _patch_generator(monkeypatch, _generator([_record(0), _record(1)], calls=resumed_calls))
    result = runtime.run_record_template(None, IDENTIFIER, **kwargs)
    assert result["records"] == [_record(0), _record(1)]
    assert len(resumed_calls) == 1
    assert len(progress) == 1
    assert next(iter(progress)).startswith("record-set-v1:")

    _patch_generator(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("completed record set regenerated"),
    )
    assert runtime.run_record_template(None, IDENTIFIER, **kwargs) == result


@pytest.mark.parametrize("changed", ["context", "allowed_refs"])
def test_changed_inputs_do_not_reuse_prior_record_set(monkeypatch, changed):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator([_record(0)]))
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


def test_changed_template_contract_invalidates_saved_record_set(monkeypatch):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator([_record(0)]))
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)

    original_loader = bounded.load_record_template

    def changed(identifier):
        template = original_loader(identifier)
        template["task"] += " Clarified instruction."
        return template

    monkeypatch.setattr(bounded, "load_record_template", changed)

    def contract_changed(*args, **kwargs):
        raise StopIteration("contract changed")

    _patch_generator(monkeypatch, contract_changed)
    with pytest.raises(StopIteration, match="contract changed"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_saved_record_set_is_revalidated_before_any_model_call(monkeypatch):
    progress: dict = {}
    _patch_generator(monkeypatch, _generator([_record(0)]))
    kwargs = dict(
        context={},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)
    key = next(iter(progress))
    progress[key]["records"][0]["trigger"] = "   "
    _patch_generator(
        monkeypatch,
        lambda *args, **kwargs: pytest.fail("invalid saved record set must fail before model call"),
    )
    with pytest.raises(Exception):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_record_set_has_no_legacy_128_record_completion_limit(monkeypatch):
    calls = []
    records = [_record(index) for index in range(129)]
    _patch_generator(monkeypatch, _generator(records, calls=calls))
    result = runtime.run_record_template(None, IDENTIFIER, context={}, allowed_refs=set())
    assert len(result["records"]) == 129
    assert len(calls) == 1
    assert "maxItems" not in calls[0]["properties"]["records"]


def test_invalid_record_set_is_never_checkpointed(monkeypatch):
    saved: list[tuple[str, object]] = []

    def generate(*args, **kwargs):
        return {"records": [{"trigger": "   ", "owner": "server"}]}

    _patch_generator(monkeypatch, generate)
    with pytest.raises(Exception):
        runtime.run_record_template(
            None,
            IDENTIFIER,
            context={},
            allowed_refs=set(),
            checkpoint=lambda *args: saved.append(args),
        )
    assert saved == []
