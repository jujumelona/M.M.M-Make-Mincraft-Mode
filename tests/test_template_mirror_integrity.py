from pathlib import Path

from minecraft_mod_ai.task_template_catalog import RUNTIME_TEMPLATE_ROOT
from minecraft_mod_ai.template_contract_validation import runtime_consumer_roots, validate_catalog


ROOT = Path(__file__).resolve().parents[1]


def test_only_package_template_authority_exists():
    assert not (ROOT / "templates").exists(), "Production templates belong in minecraft_mod_ai/templates only"
    assert RUNTIME_TEMPLATE_ROOT == ROOT / "minecraft_mod_ai" / "templates"


def test_all_catalog_contracts_and_manifest_references():
    assert validate_catalog(RUNTIME_TEMPLATE_ROOT, consumer_roots=runtime_consumer_roots())
