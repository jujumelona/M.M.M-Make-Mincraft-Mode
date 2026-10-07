from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import api as api_module
from minecraft_mod_ai import resource_asset_preflight_contract as preflight
from minecraft_mod_ai.model_adapters import base as adapter_base


def _router() -> SimpleNamespace:
    config = SimpleNamespace(
        adapter="image_diffusion",
        model_id="black-forest-labs/FLUX.2-klein-4B",
        quantization="bnb_4bit_nf4",
        base_url="",
        api_key="",
        extra={
            "lora_model_id": "Limbicnation/pixel-art-lora",
            "lora_weight_name": "pytorch_lora_weights.safetensors",
        },
    )
    return SimpleNamespace(
        profile="t4_local",
        registry=SimpleNamespace(role=lambda _profile, _role: config),
    )


def test_preflight_checks_base_model_and_lora_before_generation(monkeypatch) -> None:
    checked: list[tuple[str, str]] = []

    monkeypatch.setattr(adapter_base, "require_package", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preflight,
        "_probe_hf_file_access",
        lambda repo_id, filename: (
            checked.append((repo_id, filename))
            or {"repo_id": repo_id, "filename": filename, "access": "remote"}
        ),
    )

    receipt = preflight.validate_image_backend_access(_router())

    assert receipt["status"] == "PASS"
    assert checked == [
        ("black-forest-labs/FLUX.2-klein-4B", "model_index.json"),
        ("Limbicnation/pixel-art-lora", "pytorch_lora_weights.safetensors"),
    ]


def test_preflight_fails_closed_when_hub_access_fails(monkeypatch) -> None:
    monkeypatch.setattr(adapter_base, "require_package", lambda *_args, **_kwargs: None)

    def denied(_repo_id: str, _filename: str):
        raise preflight.ResourceAssetPreflightError("401 gated/inaccessible")

    monkeypatch.setattr(preflight, "_probe_hf_file_access", denied)

    with pytest.raises(
        preflight.ResourceAssetPreflightError,
        match="401 gated/inaccessible",
    ):
        preflight.validate_image_backend_access(_router())


def test_build_preflight_runs_before_authored_plan_compilation() -> None:
    source = inspect.getsource(api_module.CompleteModAISession.build)
    assert source.index("validate_image_backend_access(self.router)") < source.index(
        "_production_proposal(self, proposal)"
    )
