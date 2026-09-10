from __future__ import annotations

import json
from pathlib import Path

from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
from minecraft_mod_ai.artifact_materializer import (
    materialize_binary_asset,
    materialize_job_output,
    materialize_whole_file,
)
from minecraft_mod_ai.artifact_ports import PortRegistry
from minecraft_mod_ai.artifact_validators.reference import validate_item_vertical_slice
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.task_template_catalog import load_template


def test_item_vertical_slice_end_to_end(tmp_path: Path):
    mod_id = "space"
    pkg = "com.foo.space"
    pkg_path = "com/foo/space"
    main_class = "SpaceMod"

    # 1. Initialize skeleton in tmp workspace
    mod_item_ids_skeleton = (
        "package com.foo.space.registry;\n\n"
        "import net.minecraft.core.registries.Registries;\n"
        "import net.minecraft.resources.ResourceKey;\n"
        "import net.minecraft.resources.Identifier;\n"
        "import net.minecraft.world.item.Item;\n\n"
        "public final class ModItemIds {\n"
        "    private ModItemIds() {}\n\n"
        "    /* MMM:item_keys */\n"
        "}\n"
    )
    materialize_whole_file(
        tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModItemIds.java",
        mod_item_ids_skeleton,
    )

    mod_items_skeleton = (
        "package com.foo.space.registry;\n\n"
        "import net.minecraft.core.Registry;\n"
        "import net.minecraft.core.registries.BuiltInRegistries;\n"
        "import net.minecraft.world.item.Item;\n\n"
        "public final class ModItems {\n"
        "    private ModItems() {}\n\n"
        "    /* MMM:item_registry */\n\n"
        "    public static void initialize() {}\n"
        "}\n"
    )
    materialize_whole_file(
        tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModItems.java",
        mod_items_skeleton,
    )

    main_class_skeleton = (
        "package com.foo.space;\n\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "import com.foo.space.registry.ModItems;\n\n"
        "public final class SpaceMod implements ModInitializer {\n"
        "    public static final String MOD_ID = \"space\";\n\n"
        "    @Override\n"
        "    public void onInitialize() {\n"
        "        /* MMM:init */\n"
        "    }\n"
        "}\n"
    )
    materialize_whole_file(
        tmp_path / "src" / "main" / "java" / pkg_path / f"{main_class}.java",
        main_class_skeleton,
    )

    # 2. Input facts
    facts = [
        PromptFact(
            fact_id="fact_001",
            fact_type=FactType.ITEM_EXISTS,
            subject="raw_lunite",
            source_clause="raw lunite item exists",
        ),
        PromptFact(
            fact_id="fact_002",
            fact_type=FactType.ITEM_STACK_LIMIT,
            subject="raw_lunite",
            value=16,
            source_clause="stack limit is 16",
        ),
    ]

    # 3. Lower facts to jobs
    jobs = expand_facts_to_jobs(
        facts,
        mod_id=mod_id,
        package_name=pkg,
        main_class=main_class,
    )
    assert len(jobs) == 7

    # 4. Execute jobs according to port dependencies
    port_registry = PortRegistry()
    job_map = {j.job_id: j for j in jobs}

    # Dependency order:
    execution_order = [
        "raw_lunite.key",
        "raw_lunite.register_basic",
        "raw_lunite.settings_max_stack",
        "raw_lunite.initializer",
        "raw_lunite.client_item",
        "raw_lunite.model_basic",
        "raw_lunite.lang_en",
    ]

    for job_id in execution_order:
        job = job_map[job_id]
        # Check required ports are satisfied
        for req in job.requires:
            assert port_registry.get(req) is not None, f"Missing required port {req} for {job_id}"

        template = load_template(job.template_id)
        rendered = render_template(template, job.deterministic_inputs)

        # Materialize to disk
        receipt = materialize_job_output(job, rendered, base_dir=tmp_path)
        assert receipt.status == "SUCCESS"

        # Register produced ports
        for prod in job.produces:
            port_registry.register(prod, rendered)

    # 5. Verify the entire vertical slice before texture
    v1 = validate_item_vertical_slice(
        tmp_path,
        mod_id=mod_id,
        item_name="raw_lunite",
        package_name=pkg,
        main_class=main_class,
        expected_stack_limit=16,
    )
    assert v1["status"] == "PASS"
    checks = v1["details"]["checks"]
    assert checks["resource_key_declared"] is True
    assert checks["item_registered"] is True
    assert checks["stack_limit_verified"] is True
    assert checks["main_initializer_hooked"] is True
    assert checks["client_item_model_matched"] is True
    assert checks["item_model_matched"] is True
    assert checks["lang_entry_present"] is True
    assert checks["texture_asset_present"] is False
    assert v1["details"]["unresolved_asset_obligations"] == ["space:textures/item/raw_lunite.png"]

    # Check generated file contents
    items_java = (tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModItems.java").read_text(encoding="utf-8")
    assert "public static final Item RAW_LUNITE =" in items_java
    assert "BuiltInRegistries.ITEM" in items_java
    assert "ModItemIds.RAW_LUNITE_KEY" in items_java
    assert ".stacksTo(16)" in items_java

    client_item_json = json.loads(
        (tmp_path / "src" / "main" / "resources" / "assets" / mod_id / "items" / "raw_lunite.json").read_text(encoding="utf-8")
    )
    assert client_item_json["model"]["model"] == "space:item/raw_lunite"

    # 6. Supply texture asset and re-verify
    texture_path = tmp_path / "src" / "main" / "resources" / "assets" / mod_id / "textures" / "item" / "raw_lunite.png"
    materialize_binary_asset(texture_path, b"\x89PNG\r\n\x1a\nfake_texture_bytes")

    v2 = validate_item_vertical_slice(
        tmp_path,
        mod_id=mod_id,
        item_name="raw_lunite",
        package_name=pkg,
        main_class=main_class,
        expected_stack_limit=16,
    )
    assert v2["status"] == "PASS"
    assert v2["details"]["checks"]["texture_asset_present"] is True
    assert v2["details"]["unresolved_asset_obligations"] == []
