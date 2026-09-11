"""Tests for ImplementationRegistry - ensuring real content hashes, not fake ones."""

import json
from pathlib import Path

import pytest

from minecraft_mod_ai.implementation_identity import (
    ExecutorType,
    ValidatorType,
    compute_content_hash,
    compute_json_schema_hash,
    compute_source_hash,
)
from minecraft_mod_ai.implementation_registry import (
    ImplementationRegistry,
    ImplementationRegistryError,
    reset_global_registry,
)


@pytest.fixture
def registry():
    """Fresh registry for each test."""
    reset_global_registry()
    return ImplementationRegistry()


@pytest.fixture
def temp_template(tmp_path):
    """Create a temporary template file."""
    template_path = tmp_path / "test_template.yaml"
    template_content = """
# Test template
render: |
  package {{ package_name }};
  
  public class {{ main_class }} {
      public static final String MOD_ID = "{{ mod_id }}";
  }

target:
  file: "src/main/java/{{ package_path }}/{{ main_class }}.java"
  
dependencies: []
"""
    template_path.write_text(template_content, encoding="utf-8")
    return template_path


class TestImplementationRegistry:
    """Test that registry computes real hashes, not fake string hashes."""
    
    def test_register_template_computes_real_hash(self, registry, temp_template):
        """Template hash should be computed from actual file bytes."""
        impl = registry.register_template("test/template", temp_template)
        
        # Verify it's a real SHA-256 hash with sha256: prefix
        assert impl.content_sha256.startswith("sha256:")
        hex_part = impl.content_sha256[7:]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)
        
        # Verify it matches manual computation from same bytes
        actual_bytes = temp_template.read_bytes()
        expected_hash = compute_content_hash(actual_bytes)
        assert impl.content_sha256 == expected_hash
        
        # Verify it's NOT a fake hash of string literal
        fake_hash = compute_content_hash(b"template:test/template")
        assert impl.content_sha256 != fake_hash
    
    def test_register_python_executor_hashes_source(self, registry):
        """Python executor hash should be from actual source code."""
        
        def example_generator(design: dict) -> str:
            """Example generator function."""
            return f"// Generated from {design['name']}"
        
        impl = registry.register_python_executor("example_gen", example_generator)
        
        # Verify real hash with sha256: prefix
        assert impl.content_sha256.startswith("sha256:")
        assert len(impl.content_sha256) == 71  # "sha256:" + 64 hex chars
        
        # Verify it matches source extraction
        import inspect
        source = inspect.getsource(example_generator)
        expected_hash = compute_source_hash(source)
        assert impl.content_sha256 == expected_hash
        
        # Verify it's NOT a fake hash
        fake_hash = compute_content_hash(b"generator:example_gen")
        assert impl.content_sha256 != fake_hash
    
    def test_register_validator_hashes_source(self, registry):
        """Validator hash should be from actual validator function source."""
        
        def java_syntax_validator(code: str) -> bool:
            """Example validator."""
            return "class " in code
        
        registration = registry.register_validator(
            "java_syntax",
            java_syntax_validator,
            ValidatorType.JAVA_SYNTAX,
        )
        
        # Verify real hash with sha256: prefix
        assert registration.source_hash.startswith("sha256:")
        assert len(registration.source_hash) == 71
        
        # Verify it matches source
        import inspect
        source = inspect.getsource(java_syntax_validator)
        expected_hash = compute_source_hash(source)
        assert registration.source_hash == expected_hash
        
        # NOT fake
        fake_hash = compute_content_hash(b"validator:java_syntax")
        assert registration.source_hash != fake_hash
    
    def test_register_json_schema_canonical_hash(self, registry):
        """Schema hash should be from canonical JSON representation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
            "required": ["name"],
        }
        
        registration = registry.register_json_schema(
            "TestType",
            schema,
            "test_module.TestType",
        )
        
        # Verify real hash with sha256: prefix
        assert registration.schema_hash.startswith("sha256:")
        assert len(registration.schema_hash) == 71
        
        # Verify canonical
        expected_hash = compute_json_schema_hash(schema)
        assert registration.schema_hash == expected_hash
        
        # NOT fake
        fake_hash = compute_content_hash(b"input_schema:TestType")
        assert registration.schema_hash != fake_hash
    
    def test_duplicate_registration_same_hash_is_idempotent(self, registry, temp_template):
        """Re-registering with same content should return same object."""
        impl1 = registry.register_template("test/template", temp_template)
        impl2 = registry.register_template("test/template", temp_template)
        
        assert impl1 == impl2
        assert impl1.content_sha256 == impl2.content_sha256
    
    def test_duplicate_registration_different_hash_raises(self, registry, tmp_path):
        """Re-registering same ID with different content should fail."""
        template1 = tmp_path / "template1.yaml"
        template1.write_text("content: version1")
        
        template2 = tmp_path / "template2.yaml"
        template2.write_text("content: version2")
        
        registry.register_template("test/template", template1)
        
        with pytest.raises(ImplementationRegistryError, match="already registered with different hash"):
            registry.register_template("test/template", template2)
    
    def test_hash_verification(self, registry, temp_template):
        """Verify implementation hash matches expected."""
        impl = registry.register_template("test/template", temp_template)
        
        # Correct hash verifies
        assert registry.verify_implementation_hash("test/template", impl.content_sha256)
        
        # Wrong hash fails
        assert not registry.verify_implementation_hash("test/template", "wrong_hash")
    
    def test_serialization_roundtrip(self, registry, temp_template):
        """Registry should serialize and deserialize without losing data."""
        # Register various items
        impl = registry.register_template("test/template", temp_template)
        
        def validator_fn(x): return True
        val_reg = registry.register_validator("test_val", validator_fn, ValidatorType.CUSTOM)
        
        schema_reg = registry.register_json_schema(
            "TestType",
            {"type": "string"},
            "test.TestType",
        )
        
        # Serialize
        data = registry.to_dict()
        
        # Deserialize
        new_registry = ImplementationRegistry.from_dict(data)
        
        # Verify all items preserved
        new_impl = new_registry.get_implementation("test/template")
        assert new_impl.content_sha256 == impl.content_sha256
        assert new_impl.executor_type == impl.executor_type
        
        new_val = new_registry.get_validator("test_val")
        assert new_val.source_hash == val_reg.source_hash
        
        new_schema = new_registry.get_schema("TestType")
        assert new_schema.schema_hash == schema_reg.schema_hash
    
    def test_template_content_change_produces_different_hash(self, registry, tmp_path):
        """Changing template content should produce different hash."""
        template_path = tmp_path / "template.yaml"
        
        # Version 1
        template_path.write_text("content: version1")
        impl1 = registry.register_template("template_v1", template_path)
        
        # Version 2
        template_path.write_text("content: version2")
        impl2 = registry.register_template("template_v2", template_path)
        
        # Different hashes
        assert impl1.content_sha256 != impl2.content_sha256


class TestHashComputationFunctions:
    """Test the hash computation utility functions."""
    
    def test_compute_content_hash_deterministic(self):
        """Same content should always produce same hash."""
        content = b"test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2
    
    def test_compute_source_hash_normalizes_line_endings(self):
        """Source hash should normalize line endings."""
        source_lf = "line1\nline2\nline3"
        source_crlf = "line1\r\nline2\r\nline3"
        source_cr = "line1\rline2\rline3"
        
        hash_lf = compute_source_hash(source_lf)
        hash_crlf = compute_source_hash(source_crlf)
        hash_cr = compute_source_hash(source_cr)
        
        # All should produce same hash after normalization
        assert hash_lf == hash_crlf == hash_cr
    
    def test_compute_json_schema_hash_canonical(self):
        """Schema hash should be order-independent (canonical JSON)."""
        schema1 = {"type": "object", "properties": {"a": {}, "b": {}}}
        schema2 = {"properties": {"b": {}, "a": {}}, "type": "object"}
        
        hash1 = compute_json_schema_hash(schema1)
        hash2 = compute_json_schema_hash(schema2)
        
        # Same hash despite different key order
        assert hash1 == hash2
    
    def test_compute_json_schema_hash_sensitive_to_content(self):
        """Schema hash should change when content changes."""
        schema1 = {"type": "string"}
        schema2 = {"type": "integer"}
        
        hash1 = compute_json_schema_hash(schema1)
        hash2 = compute_json_schema_hash(schema2)
        
        assert hash1 != hash2


class TestRegressionPreventionForP0_1:
    """Tests specifically preventing regression of P0-1 issue.
    
    These tests ensure we never go back to fake hashes.
    """
    
    def test_implementation_hash_not_from_string_id(self, registry, temp_template):
        """REGRESSION: Implementation hash must NOT be hash(f'generator:{id}')."""
        impl = registry.register_template("fabric/item/register_basic", temp_template)
        
        # This is what we DON'T want (fake hash)
        fake_hash = compute_content_hash(b"generator:fabric/item/register_basic")
        
        # Our hash must be different
        assert impl.content_sha256 != fake_hash
        
        # Our hash must be from actual content
        actual_content = temp_template.read_bytes()
        real_hash = compute_content_hash(actual_content)
        assert impl.content_sha256 == real_hash
    
    def test_validator_hash_not_from_string_id(self, registry):
        """REGRESSION: Validator hash must NOT be hash('validator:{id}')."""
        
        def my_validator(x): return True
        
        reg = registry.register_validator("java_syntax", my_validator, ValidatorType.JAVA_SYNTAX)
        
        # Fake hash we're avoiding
        fake_hash = compute_content_hash(b"validator:java_syntax")
        
        # Must be different
        assert reg.source_hash != fake_hash
        
        # Must be from actual source
        import inspect
        real_hash = compute_source_hash(inspect.getsource(my_validator))
        assert reg.source_hash == real_hash
    
    def test_schema_hash_not_from_string_type_name(self, registry):
        """REGRESSION: Schema hash must NOT be hash('input_schema:{type}')."""
        schema = {"type": "object", "properties": {"field": {"type": "string"}}}
        
        reg = registry.register_json_schema("EntityRenderDesign", schema, "test.EntityRenderDesign")
        
        # Fake hashes we're avoiding
        fake_hash_input = compute_content_hash(b"input_schema:EntityRenderDesign")
        fake_hash_output = compute_content_hash(b"output_schema:EntityRenderDesign")
        
        # Must be different
        assert reg.schema_hash != fake_hash_input
        assert reg.schema_hash != fake_hash_output
        
        # Must be from actual schema
        real_hash = compute_json_schema_hash(schema)
        assert reg.schema_hash == real_hash
    
    def test_evidence_id_not_just_string_concat(self):
        """REGRESSION: Evidence ID must reference actual evidence object, not just string."""
        # This test documents the requirement; implementation in evidence_store.py
        
        # WRONG (current):
        # evidence_id = f"evidence:host:{leaf}:{minecraft}"
        
        # RIGHT (required):
        # evidence_id = sha256(evidence_record_canonical_bytes)
        # where evidence_record contains actual hashes and results
        
        # Placeholder for future evidence_store test
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
