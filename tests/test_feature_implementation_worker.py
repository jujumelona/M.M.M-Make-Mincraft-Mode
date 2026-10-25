"""The small model never owns novel source or source compiler decisions."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.feature_implementation_worker import (
    FeatureImplementationRouter,
    FeatureImplementationUnavailable,
    require_feature_implementation_router,
)


class Registry:
    def __init__(self, remote=True, same=False):
        self.remote, self.same = remote, same

    def role(self, profile, role):
        if role == "feature_coder":
            if not self.remote:
                raise KeyError(role)
            return SimpleNamespace(
                adapter="openai_compatible", provider="openai_compatible",
                model_id="mimo" if self.same else "large-independent-coder",
                base_url="http://example/v1" if not self.same else "",
                api_key="not-a-real-secret",
            )
        return SimpleNamespace(model_id="mimo", base_url="")


class Router:
    profile = "t4_local"

    def __init__(self, registry):
        self.registry = registry
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters, **kwargs):
        self.calls.append(("structured", role, data_name))
        return {"choice": "ok"}

    def generate_java_source(self, role, messages, *, data_name, source_frame, **kwargs):
        self.calls.append(("java", role, data_name))
        return source_frame["prefix"] + "return true;" + source_frame["suffix"]


def test_novel_feature_with_only_mimo_is_not_sent_to_small_coder():
    router = Router(Registry(remote=False))
    with pytest.raises(FeatureImplementationUnavailable, match="FEATURE_CODER_NOT_CONFIGURED"):
        require_feature_implementation_router(router)
    assert router.calls == []


def test_new_feature_metadata_body_and_journal_owner_are_independent():
    router = Router(Registry())
    worker = require_feature_implementation_router(router)
    assert isinstance(worker, FeatureImplementationRouter)
    assert worker.generate_structured_data("planner", [], data_name="p", parameters={}) == {"choice": "ok"}
    assert "return true;" in worker.generate_java_source(
        "coder", [], data_name="j",
        source_frame={"prefix": "void m() {", "suffix": "}"},
    )
    assert router.calls == [
        ("structured", "feature_coder", "p"),
        ("java", "feature_coder", "j"),
    ]


def test_worker_does_not_relabel_unrelated_roles():
    router = Router(Registry())
    worker = require_feature_implementation_router(router)
    worker.generate_structured_data("embedding", [], data_name="e", parameters={})
    assert router.calls == [("structured", "embedding", "e")]


def test_no_hidden_remote_provider_or_secret_in_message():
    router = Router(Registry(same=True))
    with pytest.raises(FeatureImplementationUnavailable, match="FEATURE_CODER_UNVERIFIED_PROVIDER"):
        require_feature_implementation_router(router)


def test_wrapper_under_journal_keeps_journal_as_primary_recipient():
    underlying = Router(Registry())
    class Journal:
        router = underlying
        def generate_structured_data(self, role, messages, *, data_name, parameters, **kwargs):
            assert role == "feature_coder"
            return self.router.generate_structured_data(
                role, messages, data_name=data_name, parameters=parameters, **kwargs)
    worker = require_feature_implementation_router(Journal())
    worker.generate_structured_data("planner", [], data_name="p", parameters={})
    assert underlying.calls == [("structured", "feature_coder", "p")]
