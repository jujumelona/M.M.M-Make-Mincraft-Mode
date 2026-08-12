from __future__ import annotations

"""Install cross-cutting production contracts that must be active package-wide.

The individual contracts are intentionally kept in their own modules so they can be
unit-tested in isolation.  This bootstrap is the single integration point used by
``minecraft_mod_ai.__init__`` after the legacy/runtime bootstrap has loaded.  Every
installer is idempotent and marks the wrapped callable, so importing the package more
than once cannot stack duplicate wrappers.
"""


def install() -> None:
    # The optional Colab server cell can start before CompleteModAISession applies the
    # selected KV-cache quantization. Reuse must therefore be bound to the exact model,
    # context/output limits, baseline/MTP lane and KV-cache type rather than only mode.
    from . import colab_mtp_server as colab_server_module
    from .colab_server_config_contract import install as install_colab_server_config

    install_colab_server_config(colab_server_module)

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

    # Specialized deterministic templates still contain Fabric-1.20.1-specific API
    # code.  Guard both their public functions and the copies already imported by the
    # orchestrator so a newer target is routed to custom_java instead of receiving a
    # silently incompatible source tree.
    from . import geckolib_generator as geckolib_module
    from . import system_pack_generator as system_module
    from .platform_specialized_generator_contract import (
        install as install_specialized_generator_guards,
    )

    install_specialized_generator_guards(
        system_module=system_module,
        geckolib_module=geckolib_module,
        orchestrator_module=orchestrator_module,
    )

    # Built-in persistent systems get a real stop/start round-trip and corruption
    # recovery probe during disposable-runtime cleanup. Party/guild packs additionally
    # receive a two-client authority/reconnect probe. These typed receipts feed the
    # strict quality dimensions instead of relying on unsupported gate strings.
    from . import production_contract as production_contract_module
    from . import runtime_manager as runtime_module
    from . import system_templates_common as system_templates_module
    from .system_quality_contract import install as install_system_quality

    install_system_quality(
        templates_module=system_templates_module,
        system_module=system_module,
        production_contract_module=production_contract_module,
        runtime_module=runtime_module,
        orchestrator_module=orchestrator_module,
    )

    # Finally install the fail-closed atomic requirement/evidence/release architecture.
    # It deliberately runs after platform lowering so acceptance evidence is bound to
    # the actual implementation graph that will execute.
    from . import complete_orchestrator_services as services_module
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

    # final_architecture installs scheduler_fairness_contract late, which replaces
    # DurableWorkLedger.claim_ready. Re-apply the idempotent safety layer after that
    # replacement so the real orchestrator keeps executor-lane capacity balancing
    # *and* process-unique lease ownership/heartbeat. It also restores the shared
    # ProjectIndex-before-SUCCEEDED ordering if a late contract replaced that hook.
    from .scheduler_parallel_safety_contract import (
        install as install_scheduler_parallel_safety,
    )

    install_scheduler_parallel_safety(
        work_graph_module=work_graph_module,
        orchestrator_module=orchestrator_module,
    )

    # MCP production services are cached and can receive concurrent requests. Keep
    # expensive RAG indexing parallel across different outputs while enforcing one
    # writer for the same canonical index path, closing the _new_file/build TOCTOU.
    from . import production_tools as production_tools_module
    from .production_tool_parallel_contract import (
        install as install_production_tool_parallel_safety,
    )

    install_production_tool_parallel_safety(production_tools_module)
