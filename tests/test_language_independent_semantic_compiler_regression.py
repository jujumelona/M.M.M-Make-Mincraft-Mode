from __future__ import annotations

"""Language-neutral invariants for the host semantic/compiler boundary."""

import re

from minecraft_mod_ai.canonical_capability_ontology import (
    resolve_capabilities_from_phrase_structured,
)
from minecraft_mod_ai.evidence_first_planning import (
    _fallback_capability,
    _slug,
    _word_overlap,
    build_request_catalog,
)
from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability
from minecraft_mod_ai.requirement_catalog import _split_into_semantic_clauses


class TestHostFallbackLanguageIndependent:
    def test_ascii_and_unicode_inputs_resolve_to_host_owned_template_profiles(self) -> None:
        for statement in ("trade items", "鉱石を採掘する", "استخرج الموارد"):
            capability = _fallback_capability(statement)
            assert capability
            profile = profile_for_capability(capability)
            assert profile.capability == capability
            assert profile.template_id

    def test_unknown_non_latin_input_gets_stable_custom_semantic_id(self) -> None:
        statement = "Ξεκίνα μια γαλαξία"
        capability = _fallback_capability(statement)
        assert capability.startswith("custom.semantic_")
        assert capability == _fallback_capability(statement)


class TestValidationLanguageIndependent:
    def test_catalog_builds_for_multiple_scripts(self) -> None:
        for prompt in (
            "Mine ores and sell them at the shop",
            "鉱石を採掘してお金を稼ぐ",
            "استخرج الموارد وابن مركبة",
            "开采矿石，赚钱，交易物品",
        ):
            catalog = build_request_catalog(prompt, {})
            assert catalog["requirements"]

    def test_source_span_always_matches_original_prompt(self) -> None:
        for prompt in (
            "mine ores and earn money",
            "鉱石を採掘してお金を稼ぐ",
            "استخرج الموارد",
            "добыча ресурсов",
        ):
            catalog = build_request_catalog(prompt, {})
            for requirement in catalog["requirements"]:
                span = requirement["source_span"]
                assert span["char_start"] < span["char_end"]
                assert span["text"] == prompt[span["char_start"] : span["char_end"]]


class TestTokenizerUnicodeGeneral:
    def test_ontology_never_crashes_on_supported_unicode_text(self) -> None:
        for text in (
            "trade economy shop",
            "鉱石を採掘してお金を稼ぐ",
            "استخرج الموارد",
            "добыча ресурсов торговля",
            "trade кредит 取引 تجارة",
            "Ξεκίνα μια γαλαξία",
        ):
            result = resolve_capabilities_from_phrase_structured(text)
            assert result is not None
            assert all(node.capability_id for node in result.nodes)


class TestSlugLanguageNeutral:
    def test_slug_is_ascii_and_deterministic_for_all_scripts(self) -> None:
        for value in ("鉱石 採掘", "开采矿石", "استخرج", "добыча", "mine ores"):
            result = _slug(value)
            assert re.match(r"^[a-z0-9_]+$", result)
            assert result == _slug(value)

    def test_fallback_capability_is_deterministic_and_ascii_identifier(self) -> None:
        for statement in (
            "mine ores and sell them",
            "鉱石を採掘する",
            "استخرج الموارد",
            "добыча ресурсов",
            "开采矿石",
        ):
            result = _fallback_capability(statement)
            assert result
            assert re.match(r"^[a-z0-9_.]+$", result)
            assert result == _fallback_capability(statement)


class TestClauseSplittingLanguageNeutral:
    def test_structural_delimiters_split(self) -> None:
        assert len(_split_into_semantic_clauses("mine ores\nearn money\nbuild a spaceship")) == 3
        assert len(_split_into_semantic_clauses("mine ores,earn money,trade items")) == 3
        assert len(_split_into_semantic_clauses("mine ores; earn money; build a ship")) == 3
        assert len(_split_into_semantic_clauses("鉱石を採掘する,お金を稼ぐ,取引する")) == 3
        assert len(_split_into_semantic_clauses("استخرج الموارد,ابن المركبة")) == 2

    def test_language_conjunctions_do_not_create_hidden_boundaries(self) -> None:
        assert len(_split_into_semantic_clauses("鉱石を採掘してお金を稼いで取引する")) == 1
        assert len(_split_into_semantic_clauses("mine ores and earn money and trade items")) == 1

    def test_lists_strip_structural_markers(self) -> None:
        bullets = _split_into_semantic_clauses("• mine ores\n• earn money\n• build a spaceship")
        numbered = _split_into_semantic_clauses("1. mine ores\n2. earn money\n3. trade items")
        assert len(bullets) == 3
        assert len(numbered) == 3
        assert not any(re.match(r"^\d+\.", clause) for clause in numbered)


class TestWordOverlapUnicodeGeneral:
    def test_overlap_works_without_script_specific_branching(self) -> None:
        assert _word_overlap("mine ores", "mining ores")
        assert _word_overlap("鉱石 採掘", "鉱石 システム")
        assert _word_overlap("采矿 矿石 系统", "矿石 交易")
        assert _word_overlap("استخرج الموارد", "الموارد الطبيعية")
        assert _word_overlap("добыча ресурс торговля", "ресурс экономика")
        assert not _word_overlap("mine ores", "build spaceship")
