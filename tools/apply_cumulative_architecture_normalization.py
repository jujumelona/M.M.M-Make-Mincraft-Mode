from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _replace_region(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]


def patch_planning_pipeline() -> None:
    path = ROOT / "minecraft_mod_ai" / "planning_state_pipeline.py"
    text = path.read_text(encoding="utf-8")
    replacement = '''def _collect_research_if_needed(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None,
    checkpoint: PlanningCheckpoint | None,
) -> dict[str, Any]:
    if not _research_stage_needed(state):
        return state
    try:
        state = _transition(
            "collect_prompt_research",
            lambda: collect_planning_state_research_convergent(
                router,
                prompt,
                state,
                trace_metadata=trace_metadata,
            ),
            input_state=state,
        )
    except Exception as exc:
        _host_transition_notice("collect_prompt_research", state, exc)
    _checkpoint_state(checkpoint, state)
    return state


def _resolve_requirements_or_wait(
    router: Any,
    prompt: str,
    state: dict[str, Any],
    *,
    trace_metadata: Mapping[str, Any] | None,
    checkpoint: PlanningCheckpoint | None,
) -> tuple[dict[str, Any], bool]:
    """Resolve authored requirements; return ``waiting=True`` only for user-only unknowns."""

    if _requirements_exist(state):
        return state, False

    state = _collect_research_if_needed(
        router,
        prompt,
        state,
        trace_metadata=trace_metadata,
        checkpoint=checkpoint,
    )

    try:
        state = _transition(
            "compile_researched_requirements",
            lambda: compile_researched_requirements_convergent(router, prompt, state),
            input_state=state,
        )
    except Exception as exc:
        _host_transition_notice("compile_researched_requirements", state, exc)

    if _requirements_exist(state):
        _checkpoint_state(checkpoint, state)
        return state, False

    if _has_open_user_only_unknown(state):
        emit_root_cause(
            "planning_state_waiting_for_user_input",
            stage="planning_state",
            operation="requirement_selection",
            result="RESUMABLE",
            reason="an open user-only unknown must be supplied by the user before requirement selection",
            details=_state_summary(state),
        )
        _checkpoint_state(checkpoint, state)
        return state, True

    state = _host_add_requirement(state)
    emit_root_cause(
        "planning_state_host_requirement",
        stage="planning_state",
        operation="requirement_selection",
        result="CONTINUE",
        reason="host materialized the authored request as a canonical requirement",
        details=_state_summary(state),
    )
    _checkpoint_state(checkpoint, state)
    return state, False
'''
    text = _replace_region(
        text,
        "def _resolve_requirements_or_wait(\n",
        "def _select_detail_sections(\n",
        replacement,
    )
    path.write_text(text, encoding="utf-8")


def patch_generation_fallback() -> None:
    path = ROOT / "minecraft_mod_ai" / "generation_verifier_fallback_installation.py"
    text = path.read_text(encoding="utf-8")
    replacement = '''def _last_gradle_log(report: Any, report_dict: dict[str, Any]) -> str:
    if report.passed:
        return ""
    commands = report_dict.get("commands")
    if not isinstance(commands, list) or not commands:
        return ""
    final_command = commands[-1]
    if not isinstance(final_command, dict):
        return ""
    return _bounded_log_tail(str(final_command.get("log_path") or ""))


def _fallback_status(report: Any, last_log: str) -> tuple[str, bool]:
    toolchain_unavailable = (
        str(report.status).strip().upper() == "UNAVAILABLE"
        or _is_java_toolchain_failure(report.error, last_log)
    )
    if report.passed:
        return "PASS", toolchain_unavailable
    if toolchain_unavailable:
        return "UNAVAILABLE", True
    return "FAIL", False


def _fallback_diagnostics(
    report: Any,
    last_log: str,
    *,
    toolchain_unavailable: bool,
) -> list[dict[str, Any]]:
    if report.passed:
        return []
    message = str(report.error or "Gradle build failed.")
    if last_log:
        message += "\n\nGradle log tail:\n" + last_log
    code = "JAVA_TOOLCHAIN_UNAVAILABLE" if toolchain_unavailable else "GRADLE_BUILD_FAILED"
    return [
        {
            "severity": 1,
            "source": "gradle",
            "code": code,
            "message": message,
        }
    ]


def _environment_failure_fields(toolchain_unavailable: bool) -> dict[str, Any]:
    if not toolchain_unavailable:
        return {}
    return {
        "failure_class": "environment",
        "repairable": False,
        "code": "JAVA_TOOLCHAIN_UNAVAILABLE",
    }


def _fallback_reason(toolchain_unavailable: bool) -> str:
    if toolchain_unavailable:
        return "JDT verifier unavailable and Gradle Java toolchain unavailable"
    return "JDT verifier unavailable; pinned Gradle build used as host verifier"


def _gradle_fallback_receipt(
    runtime: Any,
    root: Path,
    *,
    runtime_module: Any,
    jdt_error: BaseException,
    gradle_runner_factory: Any | None = None,
) -> dict[str, Any]:
    """Run the pinned Gradle build and return a verifier-compatible receipt."""

    from .runner import GradleRunner

    cache_root = Path(runtime.workspace_root).expanduser().resolve() / ".cache" / "gradle"
    runner = (gradle_runner_factory or GradleRunner)(cache_root)
    report = runner.build(Path(root).resolve(), run_gametest=False)
    report_dict = report.to_dict()
    last_log = _last_gradle_log(report, report_dict)
    status, toolchain_unavailable = _fallback_status(report, last_log)
    diagnostics = _fallback_diagnostics(
        report,
        last_log,
        toolchain_unavailable=toolchain_unavailable,
    )
    receipt: dict[str, Any] = {
        "status": status,
        "complete": True,
        "session_id": "gradle-fallback",
        "model_id": f"gradle:{report.gradle_version}",
        "diagnostics": diagnostics,
        "error_count": len(diagnostics),
        "verifier_backend": "gradle_build",
        "fallback_from": "java_diagnostics",
        "jdt_unavailable_reason": f"{type(jdt_error).__name__}: {jdt_error}",
        "build": report_dict,
    }
    receipt.update(_environment_failure_fields(toolchain_unavailable))
    emit_root_cause(
        "generation_verifier_gradle_fallback_result",
        stage="generation",
        operation="run_gradle_build",
        gate="target_compile",
        result=status,
        reason=_fallback_reason(toolchain_unavailable),
        details={"result": receipt},
    )
    return runtime_module._bounded_result(receipt)
'''
    text = _replace_region(
        text,
        "def _gradle_fallback_receipt(\n",
        "def install() -> None:\n",
        replacement,
    )
    path.write_text(text, encoding="utf-8")


def patch_runner() -> None:
    path = ROOT / "minecraft_mod_ai" / "runner.py"
    text = path.read_text(encoding="utf-8")
    marker = "\n\nclass GradleRunner:\n"
    prepared = '''\n\n@dataclass(frozen=True)\nclass _PreparedBuild:\n    project_root: Path\n    gradle_version: str\n    gradle_sha256: str\n    gradle: Path\n    logs: Path\n    environment: dict[str, str]\n'''
    if "class _PreparedBuild:" not in text:
        text = text.replace(marker, prepared + marker, 1)

    build_replacement = '''    def _build_locked(
        self,
        project_root: Path,
        *,
        run_gametest: bool,
    ) -> BuildReport:
        prepared = self._prepare_build_context(project_root)
        if isinstance(prepared, BuildReport):
            return prepared
        return self._execute_prepared_build(prepared, run_gametest=run_gametest)

    def _prepare_build_context(self, project_root: Path) -> _PreparedBuild | BuildReport:
        project_root = project_root.resolve()
        if not (project_root / "build.gradle").is_file():
            raise BuildRunnerError(f"Not a generated Gradle project: {project_root}")
        try:
            adapter = adapter_from_project(project_root)
        except ValueError as exc:
            raise BuildRunnerError(
                f"Project platform lock is missing, mixed, or unsupported: {exc}"
            ) from exc
        gradle_version = adapter.gradle
        gradle_sha256 = adapter.gradle_sha256
        try:
            required_java = int(str(adapter.java_version).strip())
        except (TypeError, ValueError) as exc:
            raise BuildRunnerError(
                f"Project target has an invalid Java version: {adapter.java_version!r}"
            ) from exc
        if required_java <= 0:
            raise BuildRunnerError(
                f"Project target has an invalid Java version: {adapter.java_version!r}"
            )
        try:
            java_home = _resolve_project_java_home(required_java)
        except JDTWorkspaceBootstrapError as exc:
            return BuildReport(
                status="UNAVAILABLE",
                gradle_version=gradle_version,
                commands=(),
                jar_path=None,
                gametest_report=None,
                error=(
                    f"Java {required_java} toolchain unavailable for target "
                    f"{adapter.minecraft_version}: {exc}"
                ),
            )
        logs = project_root / ".minecraft_ai" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        gradle = self._ensure_gradle(gradle_version, gradle_sha256)
        environment = os.environ.copy()
        environment["GRADLE_USER_HOME"] = str(self.cache_dir / "gradle-user-home")
        environment["CI"] = "true"
        environment["JAVA_HOME"] = str(java_home)
        path_key = next((key for key in environment if key.upper() == "PATH"), "PATH")
        current_path = environment.get(path_key, "")
        java_bin = str(java_home / "bin")
        environment[path_key] = java_bin + os.pathsep + current_path if current_path else java_bin
        return _PreparedBuild(
            project_root=project_root,
            gradle_version=gradle_version,
            gradle_sha256=gradle_sha256,
            gradle=gradle,
            logs=logs,
            environment=environment,
        )

    def _execute_prepared_build(
        self,
        prepared: _PreparedBuild,
        *,
        run_gametest: bool,
    ) -> BuildReport:
        commands: list[CommandResult] = []
        if not self._wrapper_is_current(
            prepared.project_root,
            prepared.gradle_version,
            prepared.gradle_sha256,
        ):
            wrapper_result = self._run(
                name="wrapper",
                executable=prepared.gradle,
                arguments=(
                    "--no-daemon",
                    "wrapper",
                    "--gradle-version",
                    prepared.gradle_version,
                    "--gradle-distribution-sha256-sum",
                    prepared.gradle_sha256,
                    "--stacktrace",
                ),
                cwd=prepared.project_root,
                env=prepared.environment,
                log_path=prepared.logs / "gradle-wrapper.log",
            )
            commands.append(wrapper_result)
            if wrapper_result.exit_code != 0:
                return self._failed_build(prepared, commands, "Gradle wrapper generation failed.")
        force_clean = os.environ.get("MMM_GRADLE_FORCE_CLEAN", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        build_result = self._run(
            name="clean_build" if force_clean else "build",
            executable=prepared.gradle,
            arguments=(
                ("--no-daemon", "clean", "build", "--stacktrace")
                if force_clean
                else ("--no-daemon", "build", "--stacktrace")
            ),
            cwd=prepared.project_root,
            env=prepared.environment,
            log_path=prepared.logs / "gradle-build.log",
        )
        commands.append(build_result)
        if build_result.exit_code != 0:
            return self._failed_build(prepared, commands, "Gradle build failed.")
        if run_gametest:
            gametest_result = self._run(
                name="gametest",
                executable=prepared.gradle,
                arguments=("--no-daemon", "runGameTestServer", "--stacktrace"),
                cwd=prepared.project_root,
                env=prepared.environment,
                log_path=prepared.logs / "gradle-gametest.log",
            )
            commands.append(gametest_result)
            if gametest_result.exit_code != 0:
                return self._failed_build(
                    prepared,
                    commands,
                    "Headless Fabric GameTest failed.",
                    include_artifacts=True,
                )
        jar_path = self._find_release_jar(prepared.project_root)
        if jar_path is None:
            return self._failed_build(
                prepared,
                commands,
                "Gradle reported success but no remapped release JAR was found.",
                include_artifacts=True,
            )
        return BuildReport(
            status="PASS",
            gradle_version=prepared.gradle_version,
            commands=tuple(commands),
            jar_path=jar_path,
            gametest_report=self._gametest_report(prepared.project_root),
            error=None,
        )

    def _failed_build(
        self,
        prepared: _PreparedBuild,
        commands: list[CommandResult],
        error: str,
        *,
        include_artifacts: bool = False,
    ) -> BuildReport:
        return BuildReport(
            status="FAIL",
            gradle_version=prepared.gradle_version,
            commands=tuple(commands),
            jar_path=(
                self._find_release_jar(prepared.project_root) if include_artifacts else None
            ),
            gametest_report=(
                self._gametest_report(prepared.project_root) if include_artifacts else None
            ),
            error=error,
        )
'''
    text = _replace_region(
        text,
        "    def _build_locked(\n",
        "    @staticmethod\n    def _wrapper_is_current(\n",
        build_replacement,
    )

    run_replacement = '''    def _run(
        self,
        *,
        name: str,
        executable: Path,
        arguments: tuple[str, ...],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> CommandResult:
        from .agent_tool_runtime import _sanitize_observation
        from .root_cause_trace import emit_root_cause

        command = (str(executable), *arguments)
        started = time.monotonic()
        creation_flags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        )
        emit_root_cause(
            "gradle_command_start",
            stage="verify",
            operation=name,
            result="START",
            details={
                "command": _sanitize_observation(command),
                "cwd": str(cwd),
                "log_path": str(log_path),
                "timeout_seconds": self.command_timeout_seconds,
                "java_home": env.get("JAVA_HOME", ""),
                "gradle_user_home": env.get("GRADLE_USER_HOME", ""),
            },
        )
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            creationflags=creation_flags,
            start_new_session=(os.name != "nt"),
        )
        reader_errors: list[BaseException] = []
        reader = threading.Thread(
            target=copy_context().run,
            args=(self._copy_command_output, process, log_path, name, reader_errors),
            daemon=True,
        )
        reader.start()
        exit_code, timed_out = self._wait_for_process(process)
        reader.join(timeout=15)
        if not reader.is_alive() and process.stdout is not None:
            process.stdout.close()
        if reader.is_alive() or reader_errors:
            raise BuildRunnerError("Gradle command log could not be fully drained") from (
                reader_errors[0] if reader_errors else None
            )
        if timed_out:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    "\n[ M.M.M Make Mincraft Mode: command timed out; process tree terminated ]\n"
                )
        duration = time.monotonic() - started
        emit_root_cause(
            "gradle_command_result",
            stage="verify",
            operation=name,
            result=self._command_status(exit_code, timed_out=timed_out),
            details={
                "pid": process.pid,
                "exit_code": exit_code,
                "duration_seconds": round(duration, 3),
                "log_path": str(log_path),
            },
        )
        return CommandResult(
            name=name,
            command=command,
            exit_code=exit_code,
            duration_seconds=round(duration, 3),
            log_path=str(log_path),
            timed_out=timed_out,
        )

    @staticmethod
    def _copy_command_output(
        process: subprocess.Popen[str],
        log_path: Path,
        name: str,
        reader_errors: list[BaseException],
    ) -> None:
        from .agent_tool_runtime import _redact_text
        from .root_cause_trace import emit_root_cause

        try:
            with log_path.open("w", encoding="utf-8") as log:
                if process.stdout is None:
                    raise BuildRunnerError("Gradle output pipe is unavailable")
                private_key = False
                for raw in process.stdout:
                    if private_key:
                        if "-----END " in raw and "PRIVATE KEY-----" in raw:
                            private_key = False
                        continue
                    if "-----BEGIN " in raw and "PRIVATE KEY-----" in raw:
                        private_key = "-----END " not in raw
                        line = "[REDACTED_PRIVATE_KEY]\n"
                    else:
                        line = _redact_text(raw)
                    log.write(line)
                    log.flush()
                    emit_root_cause(
                        "gradle_command_output",
                        stage="verify",
                        operation=name,
                        result="INFO",
                        details={
                            "pid": process.pid,
                            "log_path": str(log_path),
                            "line": line.rstrip(),
                        },
                    )
        except Exception as exc:
            reader_errors.append(exc)
            emit_root_cause(
                "gradle_output_failure",
                stage="verify",
                operation=name,
                result="FAIL",
                reason=f"{type(exc).__name__}: {exc}",
                exc=exc,
            )

    def _wait_for_process(
        self,
        process: subprocess.Popen[str],
    ) -> tuple[int, bool]:
        try:
            process.wait(timeout=self.command_timeout_seconds)
            return int(process.returncode or 0), False
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            return 124, True
        except BaseException:
            _terminate_process_tree(process)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise

    @staticmethod
    def _command_status(exit_code: int, *, timed_out: bool) -> str:
        if timed_out:
            return "TIMEOUT"
        return "PASS" if exit_code == 0 else "FAIL"
'''
    text = _replace_region(
        text,
        "    def _run(\n",
        "    @staticmethod\n    def _find_release_jar(\n",
        run_replacement,
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_planning_pipeline()
    patch_generation_fallback()
    patch_runner()


if __name__ == "__main__":
    main()
