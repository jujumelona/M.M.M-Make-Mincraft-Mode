from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.small_model_concurrency_budget as concurrency


def _native_router():
    return SimpleNamespace(
        profile="test",
        registry=SimpleNamespace(
            role=lambda _profile, _role: SimpleNamespace(
                exclusive_gpu=True,
                provider="local",
                adapter="llama_cpp",
            )
        ),
    )


def _modules(*, generic_workers=6, model_workers=2, fail_capacity=False):
    central = SimpleNamespace()
    central._worker_count = lambda: generic_workers
    router = _native_router()

    def model_capacity(candidate, width):
        assert candidate is router
        # Match the real helper: it consults _worker_count() internally.
        # The hardener must avoid recursively re-entering model-capacity routing here.
        assert central._worker_count() == generic_workers
        if fail_capacity:
            raise RuntimeError("capacity unavailable")
        return min(int(width), model_workers)

    central._research_domain_worker_count = model_capacity

    def committee(candidate, prompt):
        assert candidate is router
        return {"workers": central._worker_count(), "prompt": prompt}

    def reviews(candidate, payload, *, target, reviewers):
        del payload, reviewers
        assert candidate is router
        return [{"workers": central._worker_count(), "target": target}]

    central.build_central_committee = committee
    central._parallel_reviews = reviews

    agentic = SimpleNamespace()

    def generate(game_design_module, candidate, prompt, *, media_paths=(), research, trace_metadata=None):
        del game_design_module, media_paths, trace_metadata
        assert candidate is router
        return {
            "workers": central._worker_count(),
            "prompt": prompt,
            "research": research,
        }

    agentic.generate_sectioned_game_design = generate
    return agentic, central, router


def test_provider_worker_budget_stays_generic_outside_model_scope():
    agentic, central, _router = _modules(generic_workers=6, model_workers=2)
    concurrency.harden(agentic, central)

    assert central._worker_count() == 6


def test_committee_reviewer_and_design_share_model_capacity_without_recursion():
    agentic, central, router = _modules(generic_workers=6, model_workers=2)
    concurrency.harden(agentic, central)

    assert central.build_central_committee(router, "request")["workers"] == 2
    reviews = central._parallel_reviews(
        router,
        {},
        target="design",
        reviewers=(("a", "A"), ("b", "B")),
    )
    assert reviews[0]["workers"] == 2
    design = agentic.generate_sectioned_game_design(
        object(),
        router,
        "request",
        research={},
    )
    assert design["workers"] == 2
    assert central._worker_count() == 6


def test_single_model_slot_serializes_all_model_backed_pools():
    agentic, central, router = _modules(generic_workers=8, model_workers=1)
    concurrency.harden(agentic, central)

    assert central.build_central_committee(router, "request")["workers"] == 1
    assert agentic.generate_sectioned_game_design(
        object(),
        router,
        "request",
        research={},
    )["workers"] == 1


def test_unknown_model_capacity_fails_closed_to_one_worker():
    agentic, central, router = _modules(generic_workers=8, model_workers=4, fail_capacity=True)
    concurrency.harden(agentic, central)

    assert central.build_central_committee(router, "request")["workers"] == 1
    assert central._worker_count() == 8


def test_unknown_router_never_inherits_process_global_model_capacity():
    agentic, central, router = _modules(generic_workers=8, model_workers=4)
    concurrency.harden(agentic, central)

    unknown = SimpleNamespace(profile="test", registry=None)
    assert central._research_domain_worker_count(unknown, 4) == 1
    assert central._research_domain_worker_count(router, 4) == 4
