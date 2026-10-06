from __future__ import annotations

import pytest

from minecraft_mod_ai.design_record_runtime import (
    TemplateBlocked,
    _normalize_content_domain_records,
)


def test_content_domain_duplicates_are_normalized_by_host_in_stable_order() -> None:
    records = [{
        "domains": ["item", "block", "item", "entity", "block"],
    }]

    assert _normalize_content_domain_records(records) == [{
        "domains": ["item", "block", "entity"],
    }]


def test_content_domain_normalization_rejects_empty_semantic_result() -> None:
    with pytest.raises(
        TemplateBlocked,
        match="TEMPLATE_CONTENT_DOMAINS_REQUIRED",
    ):
        _normalize_content_domain_records([{"domains": ["", ""]}])
