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
from minecraft_mod_ai.artifact_validators.reference import (
    validate_block_vertical_slice,
    validate_item_vertical_slice,
)
from minecraft_mod_ai.implementation_template_renderer import render_template
from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
from minecraft_mod_ai.task_template_catalog import load_template


def test_block_vertical_slice_end_to_end(tmp_path: Path):
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

    mod_block_ids_skeleton = (
        "package com.foo.space.registry;\n\n"
        "import net.minecraft.core.registries.Registries;\n"
        "import net.minecraft.resources.ResourceKey;\n"
        "import net.minecraft.resources.Identifier;\n"
        "import net.minecraft.world.level.block.Block;\n\n"
        "public final class ModBlockIds {\n"
        "    private ModBlockIds() {}\n\n"
        "    /* MMM:block_keys */\n"
        "}\n"
    )
    materialize_whole_file(
        tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModBlockIds.java",
        mod_block_ids_skeleton,
    )

    mod_blocks_skeleton = (
        "package com.foo.space.registry;\n\n"
        "import net.minecraft.core.Registry;\n"
        "import net.minecraft.core.registries.BuiltInRegistries;\n"
        "import net.minecraft.world.level.block.Block;\n"
        "import net.minecraft.world.level.block.state.BlockBehaviour;\n\n"
        "public final class ModBlocks {\n"
        "    private ModBlocks() {}\n\n"
        "    /* MMM:block_registry */\n\n"
        "    public static void initialize() {}\n"
        "}\n"
    )
    materialize_whole_file(
        tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModBlocks.java",
        mod_blocks_skeleton,
    )

    main_class_skeleton = (
        "package com.foo.space;\n\n"
        "import net.fabricmc.api.ModInitializer;\n"
        "import com.foo.space.registry.ModItems;\n"
        "import com.foo.space.registry.ModBlocks;\n\n"
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

    # 2. Input facts for raw_lunite item and lunite_ore block dropping raw_lunite
    facts = [
        PromptFact(
            fact_id="fact_001",
            fact_type=FactType.ITEM_EXISTS,
            subject="raw_lunite",
            source_clause="raw lunite item exists",
        ),
        PromptFact(
            fact_id="fact_002",
            fact_type=FactType.BLOCK_EXISTS,
            subject="lunite_ore",
            source_clause="lunite ore block exists",
        ),
        PromptFact(
            fact_id="fact_003",
            fact_type=FactType.BLOCK_DROP,
            subject="lunite_ore",
            object="raw_lunite",
            source_clause="lunite ore drops raw lunite",
        ),
    ]

    # 3. Lower facts to jobs
    jobs = expand_facts_to_jobs(
        facts,
        mod_id=mod_id,
        package_name=pkg,
        main_class=main_class,
    )
    # item: key, register_basic, client_item, model_basic, lang_en, initializer (6)
    # block: key, register_basic, blockstate_basic, model_cube_all, lang_en, initializer (6)
    # block_drop: block_drop (1) -> total 13 jobs
    assert len(jobs) == 13

    # 4. Execute jobs respecting port dependencies
    port_registry = PortRegistry()
    job_map = {j.job_id: j for j in jobs}

    execution_order = [
        # Item jobs first
        "raw_lunite.key",
        "raw_lunite.register_basic",
        "raw_lunite.initializer",
        "raw_lunite.client_item",
        "raw_lunite.model_basic",
        "raw_lunite.lang_en",
        # Block jobs
        "lunite_ore.key",
        "lunite_ore.register_basic",
        "lunite_ore.initializer",
        "lunite_ore.blockstate_basic",
        "lunite_ore.model_cube_all",
        "lunite_ore.lang_en",
        # Loot table job depends on lunite_ore.block_registry_id AND raw_lunite.registry_id
        "lunite_ore.block_drop",
    ]

    for job_id in execution_order:
        job = job_map[job_id]
        for req in job.requires:
            assert port_registry.get(req) is not None, f"Missing required port {req} for {job_id}"

        template = load_template(job.template_id)
        rendered = render_template(template, job.deterministic_inputs)

        receipt = materialize_job_output(job, rendered, base_dir=tmp_path)
        assert receipt.status == "SUCCESS"

        for prod in job.produces:
            port_registry.register(prod, rendered)

    # 5. Verify the item slice
    v_item = validate_item_vertical_slice(
        tmp_path,
        mod_id=mod_id,
        item_name="raw_lunite",
        package_name=pkg,
        main_class=main_class,
    )
    assert v_item["status"] == "PASS"

    # 6. Verify the block slice
    v_block = validate_block_vertical_slice(
        tmp_path,
        mod_id=mod_id,
        block_name="lunite_ore",
        package_name=pkg,
        main_class=main_class,
        expected_drop_item="raw_lunite",
    )
    assert v_block["status"] == "PASS"
    checks = v_block["details"]["checks"]
    assert checks["resource_key_declared"] is True
    assert checks["block_registered"] is True
    assert checks["main_initializer_hooked"] is True
    assert checks["blockstate_model_matched"] is True
    assert checks["block_model_matched"] is True
    assert checks["lang_entry_present"] is True
    assert checks["loot_drop_matched"] is True
    assert checks["texture_asset_present"] is False
    assert v_block["details"]["unresolved_asset_obligations"] == ["space:textures/block/lunite_ore.png"]

    # Check generated files on disk
    blocks_java = (tmp_path / "src" / "main" / "java" / pkg_path / "registry" / "ModBlocks.java").read_text(encoding="utf-8")
    assert "public static final Block LUNITE_ORE =" in blocks_java
    assert "BuiltInRegistries.BLOCK" in blocks_java
    assert "ModBlockIds.LUNITE_ORE_KEY" in blocks_java

    main_java = (tmp_path / "src" / "main" / "java" / pkg_path / f"{main_class}.java").read_text(encoding="utf-8")
    assert "ModItems.initialize();" in main_java
    assert "ModBlocks.initialize();" in main_java

    blockstate_json = json.loads(
        (tmp_path / "src" / "main" / "resources" / "assets" / mod_id / "blockstates" / "lunite_ore.json").read_text(encoding="utf-8")
    )
    assert blockstate_json["variants"][""]["model"] == "space:block/lunite_ore"

    loot_json = json.loads(
        (tmp_path / "src" / "main" / "resources" / "data" / mod_id / "loot_tables" / "blocks" / "lunite_ore.json").read_text(encoding="utf-8")
    )
    assert loot_json["type"] == "minecraft:block"
    assert loot_json["pools"][0]["entries"][0]["name"] == "space:raw_lunite"

    # 7. Supply block texture and re-verify
    texture_path = tmp_path / "src" / "main" / "resources" / "assets" / mod_id / "textures" / "block" / "lunite_ore.png"
    materialize_binary_asset(texture_path, b"\x89PNG\r\n\x1a\nblock_texture_bytes")

    v_block2 = validate_block_vertical_slice(
        tmp_path,
        mod_id=mod_id,
        block_name="lunite_ore",
        package_name=pkg,
        main_class=main_class,
        expected_drop_item="raw_lunite",
    )
    assert v_block2["status"] == "PASS"
    assert v_block2["details"]["checks"]["texture_asset_present"] is True
    assert v_block2["details"]["unresolved_asset_obligations"] == []
