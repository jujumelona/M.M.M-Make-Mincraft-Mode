from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import resource_asset_production as assets
from minecraft_mod_ai.complete_spec import AssetRequest


def _request(*, width: int = 16, height: int = 16) -> AssetRequest:
    return AssetRequest(
        asset_id="widget_icon",
        kind="item",
        visual_description="A compact pixel-art widget icon",
        render_kind="item.generated",
        subject_id="widget",
        requested_width=width,
        requested_height=height,
    )


def test_semantic_asset_request_is_bound_to_current_resource_contract() -> None:
    request = _request()
    request.validate()
    assert request.asset_id == "widget_icon"
    assert request.kind == "item"
    assert request.render_kind == "item.generated"
    assert request.subject_id == "widget"
    assert request.requested_width == 16
    assert request.requested_height == 16

    with pytest.raises(Exception, match="requested dimensions|width|height|resource policy"):
        _request(width=16, height=0).validate()


def test_resource_manifest_rejects_conflicting_exact_target_paths() -> None:
    rows = [
        {
            "asset_id": "widget_icon",
            "container": "mod",
            "textures": [
                {
                    "target_path": "src/main/resources/assets/example/textures/item/widget.png",
                    "role": "base",
                }
            ],
            "documents": [],
        },
        {
            "asset_id": "other_icon",
            "container": "mod",
            "textures": [
                {
                    "target_path": "src/main/resources/assets/example/textures/item/widget.png",
                    "role": "base",
                }
            ],
            "documents": [],
        },
    ]
    with pytest.raises(assets.AssetProductionError, match="Conflicting resource manifest path"):
        assets._validate_manifest(rows)


def test_reference_closure_resolves_generated_texture_and_model_references(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    run_root = tmp_path / "run"
    namespace = "example"

    texture = project_root / "src/main/resources/assets/example/textures/item/widget.png"
    parent_model = project_root / "src/main/resources/assets/example/models/item/base.json"
    texture.parent.mkdir(parents=True)
    parent_model.parent.mkdir(parents=True)
    texture.write_bytes(b"generated-texture")
    parent_model.write_text("{}\n", encoding="utf-8")

    rows = [
        {
            "asset_id": "widget_icon",
            "container": "mod",
            "textures": [],
            "documents": [
                {
                    "target_path": "src/main/resources/assets/example/models/item/widget.json",
                    "payload": {
                        "parent": "example:item/base",
                        "textures": {"layer0": "example:item/widget"},
                    },
                }
            ],
        }
    ]

    report = assets._validate_reference_closure(
        project_root,
        run_root,
        rows,
        namespace=namespace,
    )

    assert report["status"] == "PASS"
    assert report["checked_reference_count"] == 2
    assert {item["kind"] for item in report["references"]} == {"model", "texture"}

def test_sharded_resource_pack_validation_can_defer_zip_until_final_merge(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    run_root = tmp_path / "run"
    pack_root = run_root / "resource-pack"
    texture = pack_root / "assets/example/textures/item/widget.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"texture")

    proposal = SimpleNamespace(
        base_proposal=SimpleNamespace(
            spec=SimpleNamespace(
                platform=SimpleNamespace(resource_pack_format=34)
            )
        )
    )
    rows = [
        {
            "asset_id": "widget_icon",
            "container": "resource_pack",
            "textures": [
                {
                    "target_path": "assets/example/textures/item/widget.png",
                }
            ],
            "documents": [],
        }
    ]

    validation, archive = assets._validate_container_layout(
        proposal,
        project_root,
        run_root,
        rows,
        package_resource_pack=False,
    )

    pack = validation["standalone_resource_pack"]
    assert pack["status"] == "PASS"
    assert pack["root"] == str(pack_root.resolve())
    assert pack["zip"] == ""
    assert archive == ""
    assert not (run_root / "resource-packs/generated-resource-pack.zip").exists()
    assert (pack_root / "pack.mcmeta").is_file()



def test_production_generates_all_textures_before_first_alpha_worker(
    monkeypatch, tmp_path: Path,
):
    """A Colab asset shard must not alternate full FLUX reloads and ONNX."""
    from contextlib import contextmanager

    from minecraft_mod_ai import resource_alpha_segmentation as alpha
    from minecraft_mod_ai import resource_image_pipeline as pipeline
    from minecraft_mod_ai.model_adapters.image_diffusion import ImageGenerationConfig

    events = []
    textures = []
    for name in ("first", "second"):
        textures.append({
            "role": name,
            "prompt": name,
            "target_path": f"src/main/resources/assets/example/textures/item/{name}.png",
            "resource_contract": {"animation": {"mcmeta_required": False}},
            "alpha_policy": "cutout",
            "topology": "cutout_sprite",
            "width": 16,
            "height": 16,
        })
    rows = [{
        "asset_id": "batch-1",
        "container": "mod",
        "render_kind": "item.generated",
        "textures": textures,
        "documents": [],
    }]

    monkeypatch.setattr(assets, "_preflight", lambda proposal: None)
    monkeypatch.setattr(assets, "_validated_plan", lambda router, proposal: {"assets": rows})
    monkeypatch.setattr(assets, "_write_documents", lambda *args: [])
    monkeypatch.setattr(
        assets, "_validate_reference_closure",
        lambda *args, **kwargs: {"status": "PASS", "checked_reference_count": 0},
    )
    monkeypatch.setattr(
        assets, "_validate_container_layout",
        lambda *args, **kwargs: ({"mod_resources": {"status": "PASS"}}, ""),
    )
    monkeypatch.setattr(pipeline, "validate_model_consumers", lambda *args: None)
    monkeypatch.setattr(pipeline, "validate_texture", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(alpha, "_available_host_ram_bytes", lambda: 12 * 1024**3)
    monkeypatch.setattr(alpha, "prepare_foreground_model_isolated", lambda: None)

    monkeypatch.setattr(
        ImageGenerationConfig, "from_adapter_config",
        classmethod(lambda cls, config: SimpleNamespace(
            candidate_count=1, preferred_generation_resolution=(256, 256),
        )),
    )
    from minecraft_mod_ai.model_adapters import image_diffusion
    monkeypatch.setattr(
        image_diffusion, "release_image_pipeline_for_segmentation",
        lambda: events.append("release"),
    )

    def fake_prepare(generator, texture, **kwargs):
        events.append("prepare:" + texture["role"])
        return [({"name": texture["role"], "box": [0, 0, 16, 16]},
                 kwargs["directory"] / "source.png", True)]

    def fake_finish(texture, pending, *, output, resolution):
        events.append("finish:" + texture["role"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mock-verified-png")
        return {"sources": [{"alpha_matte": "rembg:u2netp"}]}

    monkeypatch.setattr(pipeline, "prepare_candidate_sources", fake_prepare)
    monkeypatch.setattr(pipeline, "finalize_candidate_sources", fake_finish)

    class Router:
        profile = "test"
        registry = SimpleNamespace(role=lambda *_: object())

        @contextmanager
        def image_generation_session(self, role):
            yield self

    proposal = SimpleNamespace(
        assets=(object(),),
        base_proposal=SimpleNamespace(spec=SimpleNamespace(mod_id="example")),
    )
    receipt = assets.generate_assets(
        Router(), proposal, tmp_path / "project", tmp_path / "run",
        package_resource_pack=False,
    )
    assert events == [
        "prepare:first", "prepare:second", "release",
        "finish:first", "finish:second",
    ]
    assert receipt["count"] == 2
    assert all(item["generation_evidence"]["sources"][0]["alpha_matte"] == "rembg:u2netp"
               for item in receipt["assets"])
