from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import planner_checkpoint as checkpoint_module
from minecraft_mod_ai.llama_server_runtime_tuning import _configure_prompt_cache
from minecraft_mod_ai.model_concurrency import (
    ModelExecutionBudget,
    ModelExecutionCancelled,
    run_with_model_execution_deadline,
)
from minecraft_mod_ai.planner_checkpoint import PlannerCheckpoint


def _checkpoint(tmp_path, monkeypatch, *, media_paths=(), model_id="qwen"):
    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_ENABLED", "1")
    return PlannerCheckpoint(
        prompt="Build one authenticated item with a recipe",
        media_paths=media_paths,
        model_id=model_id,
        source_revision="test-revision",
        target_version="1.21.1",
        target_loader="fabric",
    )


def test_crash_resume_keeps_structured_sections_and_completed_records(tmp_path, monkeypatch):
    first = _checkpoint(tmp_path, monkeypatch)
    sections = {"overview": {"specification": {}}}
    first.save_sections(sections)
    catalog = {"requirements": [{"requirement_id": "r1", "statement": "one item"}]}
    assert first.content_progress(catalog) == {}
    first.save_content_record("accepted-record-1", {"entity_id": "item_one"})
    assert first.path.is_file()

    restarted = _checkpoint(tmp_path, monkeypatch)
    assert restarted.load_sections() == sections
    assert restarted.content_progress(catalog) == {
        "accepted-record-1": {"entity_id": "item_one"}
    }
    graph = {"entities": [{"entity_id": "item_one"}], "modules": []}
    restarted.save_content_design(catalog, graph)
    assert _checkpoint(tmp_path, monkeypatch).load_content_design(catalog) == graph

    # A different catalog must never adopt records from the previous design.
    different_catalog = {"requirements": [{"requirement_id": "r2", "statement": "block"}]}
    assert restarted.content_progress(different_catalog) == {}
    assert restarted.load_content_design(different_catalog) is None


def test_checkpoint_fingerprint_tracks_model_and_media_contents(tmp_path, monkeypatch):
    image = tmp_path / "image.bin"
    image.write_bytes(b"first")
    old = _checkpoint(tmp_path, monkeypatch, media_paths=[image])
    old.save_sections({"overview": {"specification": {}}})

    image.write_bytes(b"second")
    changed_media = _checkpoint(tmp_path, monkeypatch, media_paths=[image])
    assert changed_media.path != old.path
    assert changed_media.load_sections() is None

    changed_model = _checkpoint(tmp_path, monkeypatch, model_id="another-model")
    assert changed_model.path != old.path


def test_corrupted_checkpoint_is_not_replayed(tmp_path, monkeypatch):
    before = _checkpoint(tmp_path, monkeypatch)
    before.save_sections({"overview": {"specification": {}}})
    before.path.write_text("{broken-json", encoding="utf-8")
    resumed = _checkpoint(tmp_path, monkeypatch)
    assert resumed.load_sections() is None
    assert resumed.content_progress({"requirements": []}) == {}


@pytest.mark.parametrize("replacing", [False, True])
def test_cancelled_staged_record_keeps_persisted_and_in_memory_progress(
    tmp_path, monkeypatch, replacing
):
    checkpoint = _checkpoint(tmp_path, monkeypatch)
    catalog = {"requirements": []}
    checkpoint.content_progress(catalog)
    checkpoint.save_content_record("accepted", {"entity_id": "existing_item"})
    before = checkpoint.path.read_bytes()
    expected = checkpoint.content_progress(catalog)
    entered = threading.Event()
    release = threading.Event()
    original_dump = checkpoint_module.json.dump

    def slow_dump(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return original_dump(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module.json, "dump", slow_dump)
    budget = ModelExecutionBudget(time.monotonic() + 30)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            run_with_model_execution_deadline,
            budget,
            checkpoint.save_content_record,
            "accepted" if replacing else "late",
            {"entity_id": "late_item"},
        )
        assert entered.wait(2)
        budget.cancel()
        release.set()
        with pytest.raises(ModelExecutionCancelled):
            future.result(timeout=2)
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert checkpoint.path.read_bytes() == before
    assert checkpoint.content_progress(catalog) == expected
    assert _checkpoint(tmp_path, monkeypatch).content_progress(catalog) == expected


def test_retired_qwen_mtp_flag_does_not_override_prompt_cache_policy(monkeypatch):
    # The retired Qwen hot-path flag must not override the explicit cache policy.
    monkeypatch.setenv("MMM_QWEN35_MTP_HOTPATH", "1")
    monkeypatch.delenv("MMM_LLAMA_PROMPT_CACHE_ENABLED", raising=False)
    config = SimpleNamespace(model_id="unsloth/Qwen3.5-9B-MTP-GGUF", extra={})
    args = ["--ctx-size", "8192", "--cache-prompt", "--cache-ram", "1024",
            "--cache-reuse", "256"]
    _configure_prompt_cache(args, config)
    assert "--cache-prompt" in args
    assert "--cache-ram" in args
    assert "--cache-reuse" in args


def test_non_mtp_preserves_prompt_cache(monkeypatch):
    monkeypatch.setenv("MMM_LLAMA_CACHE_RAM_MIB", "128")
    config = SimpleNamespace(model_id="other-model", extra={})
    args = ["--ctx-size", "8192"]
    _configure_prompt_cache(args, config)
    assert "--cache-prompt" in args
    index = args.index("--cache-ram")
    assert args[index + 1] == "128"


def test_explicit_profile_cache_opt_out_is_idempotent(monkeypatch):
    monkeypatch.delenv("MMM_QWEN35_MTP_HOTPATH", raising=False)
    monkeypatch.delenv("MMM_LLAMA_PROMPT_CACHE_ENABLED", raising=False)
    config = SimpleNamespace(
        model_id="any-future-model",
        extra={"prompt_cache_enabled": False},
    )
    args = ["--ctx-size", "8192", "--cache-prompt", "--cache-ram", "1024",
            "--cache-reuse", "256"]
    _configure_prompt_cache(args, config)
    _configure_prompt_cache(args, config)
    assert args == ["--ctx-size", "8192"]
