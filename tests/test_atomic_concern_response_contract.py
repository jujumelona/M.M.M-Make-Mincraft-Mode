from __future__ import annotations

import pytest

from minecraft_mod_ai.atomic_concern_source import (
    END_MARKER,
    INITIALIZE_MARKER,
    MEMBERS_MARKER,
    parse_concern_content,
)
from minecraft_mod_ai.custom_module_errors import CustomModuleGenerationError


def _response(members: str = "", initialize: str = "") -> str:
    return (
        f"{MEMBERS_MARKER}\n{members}\n"
        f"{INITIALIZE_MARKER}\n{initialize}\n"
        f"{END_MARKER}"
    )


@pytest.mark.parametrize(
    "wrapped",
    [
        lambda body: "Here is the implementation:\n" + body + "\nDone.",
        lambda body: "```java\n" + body + "\n```",
        lambda body: "\n\n" + body + "\n\n",
    ],
)
def test_inert_outer_wrapper_is_ignored(wrapped) -> None:
    body = _response("private static final int COST = 10;", "")

    members, initialize = parse_concern_content(
        wrapped(body),
        section="behavior_contract",
    )

    assert members == "private static final int COST = 10;"
    assert initialize == ""


def test_marker_cardinality_and_order_remain_fail_closed() -> None:
    duplicated = _response("private static final int COST = 10;", "") + "\n" + END_MARKER
    with pytest.raises(
        CustomModuleGenerationError,
        match="each host response marker must occur exactly once",
    ):
        parse_concern_content(duplicated, section="behavior_contract")

    out_of_order = (
        f"{INITIALIZE_MARKER}\n"
        f"{MEMBERS_MARKER}\n"
        f"{END_MARKER}"
    )
    with pytest.raises(
        CustomModuleGenerationError,
        match="host response markers are out of order",
    ):
        parse_concern_content(out_of_order, section="behavior_contract")


@pytest.mark.parametrize(
    "members",
    [
        "```java\nprivate static final int COST = 10;\n```",
        "package example;",
        "import net.minecraft.Foo;",
        "public class Escape {}",
        "public static void initialize() {}",
    ],
)
def test_executable_region_scope_escape_is_still_rejected(members: str) -> None:
    with pytest.raises(CustomModuleGenerationError):
        parse_concern_content(_response(members, ""), section="behavior_contract")


def test_non_integration_initialize_region_is_still_rejected() -> None:
    with pytest.raises(
        CustomModuleGenerationError,
        match="only integration concerns may add initialize",
    ):
        parse_concern_content(
            _response("", "register();"),
            section="behavior_contract",
        )
