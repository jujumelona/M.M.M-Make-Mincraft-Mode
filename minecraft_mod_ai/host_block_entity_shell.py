from __future__ import annotations

"""Host-owned Fabric 26.1/26.2 baseline block-entity registration.

This template supplies a concrete BLOCK + BlockEntityType + BlockEntity
and a ModInitializer entrypoint. It does NOT implement inventory, screen
handlers, networking or the user's requested custom mechanics. Those are
separate verified capabilities and must never inherit a PASS from this shell.
The 26.x API layout is pinned to Fabric docs reference source.
"""

import re
from typing import Any


_JAVA_FQCN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)+")
_JAVA_CLASS = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_REGISTRY_PART = re.compile(r"[a-z0-9_][a-z0-9_.-]*")


def basic_block_entity_candidate_contract(
    *,
    package_name: str,
    class_name: str,
    mod_id: str,
    subject: str,
    minecraft_version: str,
) -> dict[str, Any]:
    """Generate an actual baseline registered block entity, with no AI calls."""
    if not _JAVA_FQCN.fullmatch(package_name) or not _JAVA_CLASS.fullmatch(class_name):
        raise ValueError("HOST_BLOCK_ENTITY_JAVA_IDENTITY_INVALID")
    if not _REGISTRY_PART.fullmatch(mod_id) or not _REGISTRY_PART.fullmatch(subject):
        raise ValueError("HOST_BLOCK_ENTITY_REGISTRY_IDENTITY_INVALID")
    if not re.fullmatch(r"26\.(?:1|2)(?:\.[0-9]+)?", minecraft_version):
        raise ValueError("HOST_BLOCK_ENTITY_UNREVIEWED_API_EPOCH")
    # Do not derive behavior from prose. Only the exact approved registry
    # namespace/path is injected, and the host owns all source code.
    source = f"""package {package_name};

import com.mojang.serialization.MapCodec;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.object.builder.v1.block.entity.FabricBlockEntityTypeBuilder;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.world.level.block.BaseEntityBlock;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.RenderShape;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.entity.BlockEntityType;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.minecraft.world.level.block.state.BlockState;

// MMM:HOST_26_BLOCK_ENTITY_SHELL
// Baseline registered block and block entity only. No menu, networking,
// inventory synchronization or game-specific interactions are implied.
public final class {class_name} implements ModInitializer {{
    public static Block BLOCK;
    public static BlockEntityType<GeneratedBlockEntity> TYPE;

    @Override
    public void onInitialize() {{
        Identifier identifier = Identifier.fromNamespaceAndPath("{mod_id}", "{subject}");
        ResourceKey<Block> blockKey = ResourceKey.create(Registries.BLOCK, identifier);
        BLOCK = Registry.register(
                BuiltInRegistries.BLOCK, blockKey,
                new GeneratedBlock(BlockBehaviour.Properties.of().setId(blockKey)));
        TYPE = Registry.register(
                BuiltInRegistries.BLOCK_ENTITY_TYPE, identifier,
                FabricBlockEntityTypeBuilder.create(GeneratedBlockEntity::new, BLOCK).build());
    }}

    public static final class GeneratedBlock extends BaseEntityBlock {{
        public GeneratedBlock(BlockBehaviour.Properties properties) {{
            super(properties);
        }}

        @Override
        protected MapCodec<? extends BaseEntityBlock> codec() {{
            return simpleCodec(GeneratedBlock::new);
        }}

        @Override
        public RenderShape getRenderShape(BlockState state) {{
            return RenderShape.MODEL;
        }}

        @Override
        public BlockEntity newBlockEntity(BlockPos pos, BlockState state) {{
            return new GeneratedBlockEntity(pos, state);
        }}
    }}

    public static final class GeneratedBlockEntity extends BlockEntity {{
        public GeneratedBlockEntity(BlockPos pos, BlockState state) {{
            super(TYPE, pos, state);
        }}
    }}
}}
"""
    return {
        "render_mold": source,
        "slots": [],
        "capability": "basic_block_entity_with_owned_block",
        "entrypoint": package_name + "." + class_name,
    }


__all__ = ["basic_block_entity_candidate_contract"]
