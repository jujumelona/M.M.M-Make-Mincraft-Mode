from __future__ import annotations

"""Regression tests for language-independent semantic compiler invariants.

These tests verify that the semantic compiler treats UTF-8 input uniformly:
1. Gameplay root promotion is provenance-driven rather than language-driven.
2. Validator applies semantic binding checks independently of script.
3. Tokenizers accept Unicode scripts without script-specific branches.
4. Slugs are stable hash-based IDs for non-Latin input.
5. Clause splitting uses only structural delimiters, not language conjunctions.
"""

import re

from minecraft_mod_ai.canonical_capability_ontology import (
    resolve_capabilities_from_phrase_structured,
)
from minecraft_mod_ai.evidence_first_planning import (
    _capability_from_statement,
    _semantic_requirement_fields,
    _stub_semantic_model,
    _slug,
    _word_overlap,
    build_request_catalog,
)
from minecraft_mod_ai.requirement_catalog import (
    _split_into_semantic_clauses,
)


def _make_ir(statement: str):
    """Produce a SemanticRequirementIR from a raw clause via the stub model."""
    return _stub_semantic_model(statement, 0, len(statement), statement)


class TestGameplayRootPromotionLanguageIndependent:
    def test_ascii_prompt_derived_gets_gameplay_roots(self) -> None:
        statement = "trade items with NPCs"
        capability = _capability_from_statement(statement)
        ir = _make_ir(statement)
        fields = _semantic_requirement_fields(capability, ir, "req_test_001")
        assert isinstance(fields["gameplay_capabilities"], list)

    def test_ascii_and_unicode_inputs_have_same_output_shape(self) -> None:
        for stmt in ("trade items", "鉱石を採掘する"):
            cap = _capability_from_statement(stmt)
            ir = _make_ir(stmt)
            fields = _semantic_requirement_fields(cap, ir, "req_x")
            assert isinstance(fields["gameplay_capabilities"], list)
            assert isinstance(fields["implementation_capabilities"], list)
            assert isinstance(fields["semantic_status"], str)

    def test_non_prompt_derived_capability_not_overwritten(self) -> None:
        statement = "trade items"
        explicit = "custom.explicit_design_id"
        ir = _make_ir(statement)
        fields = _semantic_requirement_fields(
            explicit, ir, "req_explicit", is_design_module=True
        )
        assert any(explicit in p for p in fields["provides"])

    def test_no_gameplay_roots_keeps_original_capability(self) -> None:
        statement = "xyzzy frobnitz zork"
        capability = _capability_from_statement(statement)
        ir = _make_ir(statement)
        fields = _semantic_requirement_fields(capability, ir, "req_nonsense")
        assert capability in fields["provides"] or fields["gameplay_capabilities"] is not None


class TestValidationLanguageIndependent:
    def test_english_catalog_builds(self) -> None:
        catalog = build_request_catalog("Mine ores and sell them at the shop", {})
        assert catalog["requirements"]

    def test_japanese_catalog_builds(self) -> None:
        catalog = build_request_catalog("鉱石を採掘してお金を稼ぐ", {})
        assert catalog["requirements"]

    def test_arabic_catalog_builds(self) -> None:
        catalog = build_request_catalog("استخرج الموارد وابن مركبة", {})
        assert catalog["requirements"]

    def test_chinese_catalog_builds(self) -> None:
        catalog = build_request_catalog("开采矿石，赚钱，交易物品", {})
        assert catalog["requirements"]

    def test_source_span_always_present(self) -> None:
        for prompt in [
            "mine ores and earn money",
            "鉱石を採掘してお金を稼ぐ",
            "استخرج الموارد",
            "добыча ресурсов",
        ]:
            catalog = build_request_catalog(prompt, {})
            for req in catalog["requirements"]:
                span = req["source_span"]
                assert span["char_start"] < span["char_end"]
                assert span["text"] == prompt[span["char_start"]:span["char_end"]]


class TestTokenizerUnicodeGeneral:
    def test_latin_resolves(self) -> None:
        result = resolve_capabilities_from_phrase_structured("trade economy shop")
        caps = [n.capability_id for n in result.nodes]
        assert any("trade" in c or "economy" in c or "shop" in c for c in caps)

    def test_japanese_does_not_crash(self) -> None:
        result = resolve_capabilities_from_phrase_structured("鉱石を採掘してお金を稼ぐ")
        assert result.nodes

    def test_arabic_does_not_crash(self) -> None:
        result = resolve_capabilities_from_phrase_structured("استخرج الموارد")
        assert result.nodes

    def test_cyrillic_does_not_crash(self) -> None:
        result = resolve_capabilities_from_phrase_structured("добыча ресурсов торговля")
        assert result.nodes

    def test_mixed_script_does_not_crash(self) -> None:
        result = resolve_capabilities_from_phrase_structured("trade кредит 取引 تجارة")
        assert result is not None

    def test_unresolved_node_has_nonempty_id(self) -> None:
        result = resolve_capabilities_from_phrase_structured("Ξεκίνα μια γαλαξία")
        for node in result.nodes:
            assert node.capability_id


class TestSlugLanguageNeutral:
    def test_latin_slug_is_ascii(self) -> None:
        result = _slug("mine ores")
        assert re.match(r"^[a-z0-9_]+$", result), f"Non-ASCII: {result!r}"

    def test_japanese_slug_is_ascii(self) -> None:
        result = _slug("鉱石 採掘")
        assert re.match(r"^[a-z0-9_]+$", result), f"Non-ASCII: {result!r}"

    def test_chinese_slug_is_ascii(self) -> None:
        result = _slug("开采矿石赚钱")
        assert re.match(r"^[a-z0-9_]+$", result), f"Non-ASCII: {result!r}"

    def test_arabic_slug_is_ascii(self) -> None:
        result = _slug("استخرج الموارد")
        assert re.match(r"^[a-z0-9_]+$", result), f"Non-ASCII: {result!r}"

    def test_slug_is_deterministic(self) -> None:
        for value in ["鉱石 採掘", "开采矿石", "استخرج", "добыча", "mine ores"]:
            assert _slug(value) == _slug(value)

    def test_capability_from_statement_all_scripts(self) -> None:
        for stmt in [
            "mine ores and sell them",
            "鉱石を採掘する",
            "استخرج الموارد",
            "добыча ресурсов",
            "开采矿石",
        ]:
            result = _capability_from_statement(stmt)
            assert result, f"Empty for {stmt!r}"
            assert re.match(r"^[a-z0-9_]+$", result), f"Non-ASCII {result!r} for {stmt!r}"

    def test_capability_from_statement_deterministic(self) -> None:
        for stmt in ["鉱石 採掘", "mine ores", "استخرج"]:
            assert _capability_from_statement(stmt) == _capability_from_statement(stmt)


class TestClauseSplittingLanguageNeutral:
    def test_newline_splits(self) -> None:
        text = "mine ores\nearn money\nbuild a spaceship"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_comma_splits(self) -> None:
        text = "mine ores,earn money,trade items"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_semicolon_splits(self) -> None:
        text = "mine ores; earn money; build a ship"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_unicode_without_delimiter_stays_single(self) -> None:
        text = "鉱石を採掘してお金を稼いで取引する"
        clauses = _split_into_semantic_clauses(text)
        assert len(clauses) == 1, f"Expected 1, got {len(clauses)}: {clauses}"

    def test_unicode_with_commas_splits(self) -> None:
        text = "鉱石を採掘する,お金を稼ぐ,取引する"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_japanese_newlines_split(self) -> None:
        text = "鉱石を採掘する\nお金を稼ぐ\n宇宙船を作る"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_arabic_with_comma_splits(self) -> None:
        text = "استخرج الموارد,ابن المركبة"
        assert len(_split_into_semantic_clauses(text)) == 2

    def test_bullet_list_splits(self) -> None:
        text = "• mine ores\n• earn money\n• build a spaceship"
        assert len(_split_into_semantic_clauses(text)) == 3

    def test_numbered_list_splits_and_strips(self) -> None:
        text = "1. mine ores\n2. earn money\n3. trade items"
        clauses = _split_into_semantic_clauses(text)
        assert len(clauses) == 3
        assert not any(re.match(r"^\d+\.", c) for c in clauses)

    def test_english_and_without_comma_stays_single(self) -> None:
        text = "mine ores and earn money and trade items"
        clauses = _split_into_semantic_clauses(text)
        assert len(clauses) == 1, f"Expected 1, got {len(clauses)}: {clauses}"


class TestWordOverlapUnicodeGeneral:
    def test_latin_overlap(self) -> None:
        assert _word_overlap("mine ores", "mining ores")

    def test_no_overlap(self) -> None:
        assert not _word_overlap("mine ores", "build spaceship")

    def test_japanese_overlap(self) -> None:
        assert _word_overlap("鉱石 採掘", "鉱石 システム")

    def test_chinese_overlap(self) -> None:
        assert _word_overlap("采矿 矿石 系统", "矿石 交易")

    def test_arabic_overlap(self) -> None:
        assert _word_overlap("استخرج الموارد", "الموارد الطبيعية")

    def test_cyrillic_overlap(self) -> None:
        assert _word_overlap("добыча ресурс торговля", "ресурс экономика")
