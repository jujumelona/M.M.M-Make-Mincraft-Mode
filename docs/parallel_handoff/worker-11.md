REQUESTING_WORKER: 11
TARGET_OWNER: 12
WHY:
- Worker 11 confirmed a shared-orchestrator failure boundary that catches every Exception from JDT diagnostics and converts it to status=UNAVAILABLE.
- That behavior hides TypeError/AttributeError/other programming defects as external JDT availability failures, so the user sees a misleading fallback instead of the causal bug.
- CLI/shared orchestration is worker-12 owned, so worker 11 is not rewriting the top-level control flow.

REQUIRED_INTERFACE_CHANGE:
- In `CompleteProductionOrchestrator.execute`, replace the broad `except Exception` around `JavaLanguageService().diagnostics(...)` with explicit operational exceptions only (for example JDTLanguageServerError, FileNotFoundError/OSError, and the timeout class actually emitted by the JDT client).
- Let unexpected programming exceptions propagate unchanged.
- Emit/attach the canonical `minecraft_mod_ai.diagnostics.FailureEvent` fields at the shared boundary so user-facing rendering can show ROOT FAILURE / CAUSE / ATTEMPTS / FALLBACK / FINAL STATUS without repeated tracebacks.
- Apply the same rule at the CLI/top-level shared exception boundary: operational/user failures may be rendered compactly; INTERNAL failures must retain debug traceback evidence and must not be silently relabeled as availability failures.

FILES_INVOLVED:
- minecraft_mod_ai/complete_orchestrator.py
- minecraft_mod_ai/cli.py
- minecraft_mod_ai/diagnostics.py

TEST_THAT_CURRENTLY_FAILS:
- Add an integration regression where the JDT diagnostics call raises TypeError: it must propagate as an INTERNAL/programming failure, not return/raise an UNAVAILABLE JDT dependency result.
- Add an operational case (JDTLanguageServerError or missing executable): it may become UNAVAILABLE and should render one compact causal diagnostic.
