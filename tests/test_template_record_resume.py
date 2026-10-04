from copy import deepcopy

import pytest

from minecraft_mod_ai import bounded_record_template as bounded
from minecraft_mod_ai import design_record_runtime as runtime


IDENTIFIER = "feature/behavior_contract/entry_conditions"


def _record(index: int) -> dict[str, str]:
    return {"trigger": f"trigger-{index}", "owner": "server"}


def _patch_count(monkeypatch, value):
    calls = []

    def generate(*args, **kwargs):
        del args
        calls.append(kwargs["response_schema"])
        if isinstance(value, BaseException):
            raise value
        return {"count": int(value)}

    monkeypatch.setattr(bounded, "generate_fixed_template_value", generate)
    return calls


def _patch_records(monkeypatch, records, *, failure=None):
    calls = []
    rows = [deepcopy(record) for record in records]

    def generate_record(
        router,
        identifier,
        *,
        context,
        progress,
        checkpoint,
    ):
        del router, identifier, progress, checkpoint
        index = int(context["record_index"])
        calls.append(deepcopy(context))
        if failure is not None:
            raise failure
        return deepcopy(rows[index])

    monkeypatch.setattr(bounded, "run_single_record_template", generate_record)
    return calls


def test_interruption_resumes_from_host_owned_count_without_reasking(monkeypatch):
    progress: dict = {}
    count_calls = _patch_count(monkeypatch, 2)
    record_calls = _patch_records(
        monkeypatch,
        [_record(0), _record(1)],
        failure=TimeoutError("transport interrupted"),
    )
    kwargs = dict(
        context={"criterion": "A"},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )

    with pytest.raises(TimeoutError):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)

    assert len(count_calls) == 1
    assert len(record_calls) == 1
    assert len(progress) == 1
    binding = next(iter(progress))
    assert binding.startswith("record-set-v2:")
    assert progress[binding] == {"count": 2}

    _patch_count(
        monkeypatch,
        AssertionError("saved host count must not be regenerated"),
    )
    resumed_record_calls = _patch_records(monkeypatch, [_record(0), _record(1)])
    result = runtime.run_record_template(None, IDENTIFIER, **kwargs)

    assert result["records"] == [_record(0), _record(1)]
    assert [item["record_index"] for item in resumed_record_calls] == [0, 1]
    assert progress[binding] == {
        "count": 2,
        "records": [_record(0), _record(1)],
    }

    _patch_count(monkeypatch, AssertionError("completed set regenerated"))
    _patch_records(
        monkeypatch,
        [],
        failure=AssertionError("completed records regenerated"),
    )
    assert runtime.run_record_template(None, IDENTIFIER, **kwargs) == result


@pytest.mark.parametrize("changed", ["context", "allowed_refs"])
def test_changed_inputs_do_not_reuse_prior_record_set(monkeypatch, changed):
    progress: dict = {}
    _patch_count(monkeypatch, 1)
    _patch_records(monkeypatch, [_record(0)])
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

    _patch_count(monkeypatch, StopIteration("binding changed"))
    with pytest.raises(StopIteration, match="binding changed"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_changed_template_contract_invalidates_saved_record_set(monkeypatch):
    progress: dict = {}
    _patch_count(monkeypatch, 1)
    _patch_records(monkeypatch, [_record(0)])
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
    _patch_count(monkeypatch, StopIteration("contract changed"))

    with pytest.raises(StopIteration, match="contract changed"):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_saved_record_set_is_revalidated_before_any_model_call(monkeypatch):
    progress: dict = {}
    _patch_count(monkeypatch, 1)
    _patch_records(monkeypatch, [_record(0)])
    kwargs = dict(
        context={},
        allowed_refs=set(),
        progress=progress,
        checkpoint=lambda key, value: progress.update({key: deepcopy(value)}),
    )
    runtime.run_record_template(None, IDENTIFIER, **kwargs)
    key = next(item for item in progress if item.startswith("record-set-v2:"))
    progress[key]["records"][0]["trigger"] = "   "

    _patch_count(
        monkeypatch,
        AssertionError("invalid saved set must fail before count generation"),
    )
    _patch_records(
        monkeypatch,
        [],
        failure=AssertionError("invalid saved set must fail before record generation"),
    )
    with pytest.raises(Exception):
        runtime.run_record_template(None, IDENTIFIER, **kwargs)


def test_record_set_count_is_finitely_bounded_before_record_jobs(monkeypatch):
    _patch_count(monkeypatch, 17)
    _patch_records(
        monkeypatch,
        [],
        failure=AssertionError("out-of-range count must not schedule record generation"),
    )

    with pytest.raises(ValueError, match="outside 0..16"):
        runtime.run_record_template(
            None,
            IDENTIFIER,
            context={},
            allowed_refs=set(),
        )


def test_invalid_record_set_is_never_checkpointed_as_complete(monkeypatch):
    saved: list[tuple[str, object]] = []
    _patch_count(monkeypatch, 1)
    _patch_records(
        monkeypatch,
        [{"trigger": "   ", "owner": "server"}],
    )

    with pytest.raises(Exception):
        runtime.run_record_template(
            None,
            IDENTIFIER,
            context={},
            allowed_refs=set(),
            checkpoint=lambda *args: saved.append(args),
        )

    assert saved
    assert all(
        not (isinstance(value, dict) and "records" in value)
        for _key, value in saved
    )
