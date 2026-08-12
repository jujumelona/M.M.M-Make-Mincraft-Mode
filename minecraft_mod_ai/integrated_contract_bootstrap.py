from __future__ import annotations

"""Install cross-cutting production contracts that must be active package-wide.

The individual contracts are intentionally kept in their own modules so they can be
unit-tested in isolation.  This bootstrap is the single integration point used by
``minecraft_mod_ai.__init__`` after the legacy/runtime bootstrap has loaded.  Every
installer is idempotent and marks the wrapped callable, so importing the package more
than once cannot stack duplicate wrappers.
"""


def install() -> None:
    # Platform generation/validation must agree with the exact approved adapter.
    from . import generator as generator_module
    from . import validator as validator_module
    from .platform_generation_contract import install as install_platform_generation
    from .platform_validation_contract import install as install_platform_validation

    install_platform_generation(generator_module)
    install_platform_validation(validator_module)

    # Planning and implementation must use the same target-scoped evidence.  The
    # central live-target compiler is installed after the planning contract so it is
    # the outer lowering pass for future Fabric targets.
    from . import central_research as central_research_module
    from . import complete_planner as complete_planner_module
    from . import game_design as game_design_module
    from . import retrieval as retrieval_module
    from . import technology_radar as technology_module
    from .platform_planning_contract import install as install_platform_planning
    from .platform_central_ai_contract import install as install_platform_central_ai

    install_platform_planning(
        game_design_module=game_design_module,
        complete_planner_module=complete_planner_module,
        central_research_module=central_research_module,
        retrieval_module=retrieval_module,
        technology_module=technology_module,
    )
    install_platform_central_ai(
        game_design_module=game_design_module,
        complete_planner_module=complete_planner_module,
    )

    # Direct custom generation/repair tools must never fall back to a stale 1.20.1
    # prompt when the project is bound to another platform.
    from . import custom_module_generator as custom_module_generator_module
    from . import repair_engine as repair_module
    from .platform_custom_coder_contract import install as install_platform_custom_coder
    from .platform_repair_target_contract import install as install_platform_repair

    install_platform_custom_coder(custom_module_generator_module)
    install_platform_repair(repair_module)

    # platform_runtime_contract is installed by __init__ immediately before this
    # bootstrap.  Wrap the resulting project preparation path so live-discovered
    # targets use the official Fabric template rather than the legacy generator.
    from . import complete_orchestrator as orchestrator_module
    from .platform_live_execution_contract import install as install_live_execution

    install_live_execution(orchestrator_module)

    # Finally install the fail-closed atomic requirement/evidence/release architecture.
    # It deliberately runs after platform lowering so acceptance evidence is bound to
    # the actual implementation graph that will execute.
    from . import complete_orchestrator_services as services_module
    from . import production_contract as production_contract_module
    from . import quality_evidence as quality_module
    from . import validation_execution_contract as validation_module
    from . import work_graph as work_graph_module
    from .final_architecture_contract import install as install_final_architecture

    install_final_architecture(
        game_design_module=game_design_module,
        complete_planner_module=complete_planner_module,
        work_graph_module=work_graph_module,
        production_contract_module=production_contract_module,
        orchestrator_module=orchestrator_module,
        services_module=services_module,
        quality_module=quality_module,
        repair_module=repair_module,
        validation_module=validation_module,
    )
