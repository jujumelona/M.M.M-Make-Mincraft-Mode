from __future__ import annotations

from minecraft_mod_ai.model_router import _usable_rag_result


def test_nested_unscored_workspace_rag_hits_are_usable() -> None:
    value = {
        "structured_content": {
            "schema_version": "mmm/code-rag-result-v1",
            "query": "DebugToken.java DebugToken class definition",
            "hits": [
                {
                    "path": "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureMod.java",
                    "source_path": "src/main/java/dev/mmm/debugfixture/MmmDebugFixtureMod.java",
                    "text": (
                        "package dev.mmm.debugfixture;\n"
                        "import net.fabricmc.api.ModInitializer;\n"
                        "public final class MmmDebugFixtureMod implements ModInitializer {}\n"
                    ),
                    "sha256": "sha256:fixture",
                    "metadata": {
                        "minecraft_version": "26.2",
                        "loader": "fabric",
                        "java_version": "25",
                        "mapping_namespace": "official",
                    },
                }
            ],
            "receipt": {
                "status": "FOUND",
                "result_count": 1,
            },
        },
        "text": [],
    }

    assert _usable_rag_result(value) is True


def test_nested_unscored_receipt_without_concrete_hits_is_not_usable() -> None:
    value = {
        "structured_content": {
            "schema_version": "mmm/code-rag-result-v1",
            "hits": [],
            "receipt": {
                "status": "FOUND",
                "result_count": 1,
            },
        }
    }

    assert _usable_rag_result(value) is False
