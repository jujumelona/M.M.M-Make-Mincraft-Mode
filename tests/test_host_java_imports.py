"""Host-owned import resolution must never guess uninspected Fabric APIs."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.host_java_imports import host_add_verified_imports


def inventory(*names):
    return {name: {"kind": "CLASS", "owner": name, "name": name.split(".")[-1],
                   "access": 1, "side": "COMMON"} for name in names}


def test_imports_exact_inspected_class_from_java_ast():
    code = ("package example;\npublic class Demo {\n"
            "  EntityType<?> type;\n}\n")
    result = host_add_verified_imports(
        code, symbols=inventory("net.minecraft.world.entity.EntityType"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import net.minecraft.world.entity.EntityType;" in result
    assert "EntityType<?> type;" in result


def test_ambiguous_simple_name_is_rejected_not_guessed():
    code = "package example;\npublic class Demo { Registry registry; }\n"
    symbols = inventory("net.minecraft.core.Registry", "java.rmi.registry.Registry")
    with pytest.raises(ValueError, match="JAVA_HOST_IMPORT_AMBIGUOUS: Registry"):
        host_add_verified_imports(
            code, symbols=symbols, owner="example.Demo",
            target_path="src/main/java/example/Demo.java")


def test_private_or_client_only_class_is_not_imported_in_common():
    code = "package example;\npublic class Demo { GameRenderer engine; }\n"
    symbols = inventory("net.minecraft.client.renderer.GameRenderer")
    result = host_add_verified_imports(
        code, symbols=symbols, owner="example.Demo",
        target_path="src/main/java/example/Demo.java")
    assert "import net.minecraft.client" not in result


def test_same_package_and_java_lang_do_not_require_import():
    code = "package example;\npublic class Demo { User user; String text; }\n"
    result = host_add_verified_imports(
        code, symbols=inventory("example.User"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import " not in result


def test_model_imports_are_rejected():
    code = ("package example;\nimport java.util.List;\n"
            "public class Demo { List<String> names; }\n")
    with pytest.raises(ValueError, match="JAVA_HOST_IMPORTS_MUST_BE_HOST_OWNED"):
        host_add_verified_imports(
            code, symbols={}, owner="example.Demo",
            target_path="src/main/java/example/Demo.java")


def test_standard_collection_import_from_jdk():
    code = "package example;\npublic class Demo { List<String> names; }\n"
    result = host_add_verified_imports(
        code, symbols={}, owner="example.Demo",
        target_path="src/main/java/example/Demo.java")
    assert "import java.util.List;" in result


def test_static_registry_reference_requires_verified_import():
    code = ("package example;\npublic class Demo {\n"
            " void init() { Registry.register(null, null, null); }\n}\n")
    result = host_add_verified_imports(
        code, symbols=inventory("net.minecraft.core.Registry"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import net.minecraft.core.Registry;" in result


def test_nested_type_reference_does_not_import_unrelated_builder():
    code = ("package example;\npublic class Demo {\n"
            "  EntityType.Builder<?> builder;\n}\n")
    result = host_add_verified_imports(
        code, symbols=inventory("net.minecraft.world.entity.EntityType",
            "java.example.Builder", "net.minecraft.world.entity.EntityType.Builder"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import net.minecraft.world.entity.EntityType;" in result
    assert "import java.example.Builder;" not in result


def test_child_package_is_not_same_package_and_needs_import():
    code = "package example;\npublic class Demo { Widget field; }\n"
    result = host_add_verified_imports(
        code, symbols=inventory("example.generated.Widget"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import example.generated.Widget;" in result


def test_sibling_package_is_not_same_package_and_needs_import():
    code = "package example.generated;\npublic class Demo { Widget field; }\n"
    result = host_add_verified_imports(
        code, symbols=inventory("example.Widget"),
        owner="example.generated.Demo",
        target_path="src/main/java/example/generated/Demo.java")
    assert "import example.Widget;" in result


def test_static_method_reference_is_resolved_via_verified_owner():
    code = ("package example;\npublic class Demo {\n"
            "  java.util.function.Supplier<Object> factory = Widget::create;\n}\n")
    result = host_add_verified_imports(
        code, symbols=inventory("example.generated.Widget"),
        owner="example.Demo", target_path="src/main/java/example/Demo.java")
    assert "import example.generated.Widget;" in result


def test_same_package_reference_does_not_import_owner():
    code = ("package example.generated;\npublic class Demo {\n"
            "  java.util.function.Supplier<Object> factory = Widget::create;\n}\n")
    result = host_add_verified_imports(
        code, symbols=inventory("example.generated.Widget"),
        owner="example.generated.Demo",
        target_path="src/main/java/example/generated/Demo.java")
    assert "import example.generated.Widget;" not in result
