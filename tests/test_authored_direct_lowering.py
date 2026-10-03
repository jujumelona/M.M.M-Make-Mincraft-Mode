import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules


def test_untyped_authored_plan_is_rejected_before_production() -> None:
    text = (
        "# behavior_contract\nTrade ore.\n"
        "# state_model\nPlayerCredits, Ship, ShipPart, persistence.\n"
        "# verification\nTest transactions."
    )
    plan = AuthoredPlan("space economy", text)

    with pytest.raises(ValueError, match="TYPED_PLAN_REQUIRED"):
        _compile_new_authored_modules(
            plan,
            mod_id="authored_test",
            package_name="example",
            target={
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "mappings": "none",
            },
        )
