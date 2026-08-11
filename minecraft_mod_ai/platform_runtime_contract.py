from __future__ import annotations

import json
import os
import re
import subprocess
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

from .platform_catalog import adapter_for_lock_values, adapter_for_target, adapter_from_project


_ACTIVE_ADAPTER: ContextVar[Any | None] = ContextVar(
    "mmm_runtime_platform_adapter", default=None
)


def install(
    *,
    orchestrator_module: Any,
    runtime_manager_module: Any,
    mineflayer_module: Any,
) -> None:
    _install_runtime_manager(runtime_manager_module)
    _install_mineflayer_target(mineflayer_module)
    _install_orchestrator_runtime(orchestrator_module)


def _install_runtime_manager(module: Any) -> None:
    cls = module.MinecraftRuntimeManager
    if getattr(cls.__init__, "_mmm_dynamic_platform_runtime", False):
        return

    def init(
        self: Any,
        workspace_root: str | Path,
        *,
        profile_name: str = "fabric_target_disposable",
        config_path: str | Path | None = None,
    ) -> None:
        import threading
        import yaml

        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        path = (
            Path(config_path).expanduser().resolve()
            if config_path is not None
            else module.resolve_config_path("runtime_profiles.yaml")
        )
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "mmm/runtime-profiles-v1":
            raise module.RuntimePolicyError("Unsupported runtime profile registry.")
        profiles = raw.get("profiles", {})
        entry = profiles.get(profile_name) if isinstance(profiles, dict) else None
        if not isinstance(entry, dict):
            # Backwards-compatible config fallback for the checked-in legacy profile.
            legacy = profiles.get("fabric_1201_disposable") if isinstance(profiles, dict) else None
            if profile_name == "fabric_target_disposable" and isinstance(legacy, dict):
                entry = legacy
                profile_name = "fabric_1201_disposable"
            else:
                raise module.RuntimePolicyError(f"Unknown runtime profile: {profile_name}")

        minecraft_version = str(entry.get("minecraft_version", "")).strip()
        loader = str(entry.get("loader", "fabric")).strip().lower()
        try:
            adapter = adapter_for_target(minecraft_version, loader)
        except ValueError as exc:
            raise module.RuntimePolicyError(str(exc)) from exc
        java_project_version = str(entry.get("java_project_version", "")).strip()
        if java_project_version != adapter.java_version:
            raise module.RuntimePolicyError(
                f"Runtime profile Java target {java_project_version!r} does not match "
                f"{adapter.adapter_id} Java {adapter.java_version}."
            )
        if not bool(entry.get("disposable_only")):
            raise module.RuntimePolicyError("MMM runtime profiles must be disposable-only.")

        self.profile = module.RuntimeProfile(
            name=profile_name,
            minecraft_version=minecraft_version,
            server_java_command=str(entry["server_java_command"]),
            server_memory_mb=int(entry["server_memory_mb"]),
            server_launcher_relative=str(entry["server_launcher_relative"]),
            client_command_env=str(entry["client_command_env"]),
            allowed_server_commands=tuple(
                re.compile(pattern) for pattern in entry["allowed_server_commands"]
            ),
            startup_ready_patterns=tuple(
                re.compile(pattern) for pattern in entry["startup_ready_patterns"]
            ),
            disposable_only=True,
            eula_must_be_explicitly_accepted=bool(
                entry["eula_must_be_explicitly_accepted"]
            ),
        )
        self._mmm_platform_adapter = adapter
        self.server_process = None
        self.client_process = None
        self.instance_root = None
        self._server_log = []
        self._client_log = []
        self._lock = threading.RLock()

    init._mmm_dynamic_platform_runtime = True
    cls.__init__ = init

    original_prepare = cls.prepare_instance
    @wraps(original_prepare)
    def prepare_instance(self: Any, *args: Any, **kwargs: Any):
        _validate_java_command(
            self.profile.server_java_command,
            expected_major=int(self._mmm_platform_adapter.java_version),
            error_type=module.RuntimePolicyError,
        )
        result = original_prepare(self, *args, **kwargs)
        result = dict(result)
        result["platform_adapter"] = self._mmm_platform_adapter.adapter_id
        result["java_version"] = self._mmm_platform_adapter.java_version
        return result

    prepare_instance._mmm_dynamic_platform_runtime = True
    cls.prepare_instance = prepare_instance


def _install_mineflayer_target(module: Any) -> None:
    cls = module.MineflayerBridge
    original_start = cls.start
    if getattr(original_start, "_mmm_dynamic_platform_runtime", False):
        return

    @wraps(original_start)
    def start(self: Any) -> None:
        adapter = _ACTIVE_ADAPTER.get()
        if adapter is None:
            version = os.environ.get("MMM_MINEFLAYER_MC_VERSION", "1.20.1").strip()
            try:
                adapter = adapter_for_target(version, "fabric")
            except ValueError as exc:
                raise module.MineflayerBridgeError(str(exc)) from exc
        os.environ["MMM_MINEFLAYER_MC_VERSION"] = adapter.minecraft_version
        return original_start(self)

    start._mmm_dynamic_platform_runtime = True
    cls.start = start


def _install_orchestrator_runtime(module: Any) -> None:
    cls = module.CompleteProductionOrchestrator
    original_execute = cls.execute
    if getattr(original_execute, "_mmm_dynamic_platform_runtime", False):
        return

    @wraps(original_execute)
    def execute(self: Any, proposal: Any, *args: Any, **kwargs: Any):
        parsed = (
            proposal
            if isinstance(proposal, module.CompleteProposal)
            else module.CompleteProposal.from_dict(proposal)
        )
        adapter = adapter_for_lock_values(parsed.base_proposal.spec.platform)
        token = _ACTIVE_ADAPTER.set(adapter)
        previous = os.environ.get("MMM_MINEFLAYER_MC_VERSION")
        os.environ["MMM_MINEFLAYER_MC_VERSION"] = adapter.minecraft_version
        try:
            return original_execute(self, parsed, *args, **kwargs)
        finally:
            _ACTIVE_ADAPTER.reset(token)
            if previous is None:
                os.environ.pop("MMM_MINEFLAYER_MC_VERSION", None)
            else:
                os.environ["MMM_MINEFLAYER_MC_VERSION"] = previous

    execute._mmm_dynamic_platform_runtime = True
    cls.execute = execute

    def runtime_profile(run_root: Path, memory_mb: int) -> Path:
        adapter = _ACTIVE_ADAPTER.get()
        if adapter is None:
            raise module.CompleteProductionError(
                "Runtime profile cannot be created without an approved platform target."
            )
        path = run_root / "integration-inputs/runtime-profile.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "mmm/runtime-profiles-v1",
            "profiles": {
                "fabric_target_disposable": {
                    "minecraft_version": adapter.minecraft_version,
                    "loader": adapter.loader,
                    "platform_adapter": adapter.adapter_id,
                    "java_project_version": int(adapter.java_version),
                    "server_java_command": _java_command_for(adapter.java_version),
                    "server_memory_mb": memory_mb,
                    "server_launcher_relative": "runtime/fabric-server-launch.jar",
                    "client_command_env": "MMM_MINECRAFT_CLIENT_COMMAND_JSON",
                    "allowed_server_commands": [
                        "^list$",
                        "^stop$",
                        "^say [A-Za-z0-9 _.,!?-]{1,120}$",
                        "^gametest runall$",
                        "^tp testplayer -?[0-9]{1,7} -?[0-9]{1,7} -?[0-9]{1,7}$",
                        "^give testplayer [a-z0-9_.-]+:[a-z0-9_./-]+( [1-9][0-9]{0,3})?$",
                    ],
                    "startup_ready_patterns": [
                        "Done \\([0-9.]+s\\)! For help, type",
                        "For help, type \\\"help\\\"",
                    ],
                    "disposable_only": True,
                    "eula_must_be_explicitly_accepted": True,
                }
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    runtime_profile._mmm_dynamic_platform_runtime = True
    cls._runtime_profile = staticmethod(runtime_profile)

    original_prepare = cls._prepare_project
    @wraps(original_prepare)
    def prepare_project(self: Any, approved: Any, *, run_root: Path, existing_input: Any):
        expected = adapter_for_lock_values(approved.base_proposal.spec.platform)
        if existing_input is not None:
            report = module.inspect_existing_project_archive(existing_input)
            if report.minecraft_version and report.minecraft_version != expected.minecraft_version:
                raise module.CompleteProductionError(
                    "Approved Revise target does not match the imported project: "
                    f"plan={expected.minecraft_version}, existing={report.minecraft_version}."
                )
            if report.loader and report.loader != expected.loader:
                raise module.CompleteProductionError(
                    "Approved Revise loader does not match the imported project: "
                    f"plan={expected.loader}, existing={report.loader}."
                )
        root = original_prepare(
            self,
            approved,
            run_root=run_root,
            existing_input=existing_input,
        )
        try:
            actual = adapter_from_project(root)
        except ValueError as exc:
            if existing_input is not None:
                raise module.CompleteProductionError(
                    "Imported project does not resolve to one reviewed exact target: " + str(exc)
                ) from exc
            raise
        if actual.adapter_id != expected.adapter_id:
            raise module.CompleteProductionError(
                f"Prepared project target {actual.adapter_id} does not match approved "
                f"target {expected.adapter_id}."
            )
        return root

    prepare_project._mmm_dynamic_platform_runtime = True
    cls._prepare_project = prepare_project

    original_matches = cls._project_matches_spec
    def project_matches_spec(project_root: Path, spec: Any) -> bool:
        if not original_matches(project_root, spec):
            return False
        try:
            return (
                adapter_from_project(project_root).adapter_id
                == adapter_for_lock_values(spec.platform).adapter_id
            )
        except Exception:
            return False

    project_matches_spec._mmm_dynamic_platform_runtime = True
    cls._project_matches_spec = staticmethod(project_matches_spec)


def _java_command_for(java_version: str) -> str:
    env_name = f"MMM_JAVA_{java_version}_CMD"
    configured = os.environ.get(env_name, "").strip()
    return configured or "java"


def _validate_java_command(command: str, *, expected_major: int, error_type: type[Exception]) -> None:
    try:
        completed = subprocess.run(
            [command, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except Exception as exc:
        raise error_type(f"Java runtime command failed: {command}: {exc}") from exc
    text = completed.stdout or ""
    match = re.search(r'version\s+"(?P<version>\d+)', text)
    if match is None:
        raise error_type(f"Could not determine Java runtime version from {command!r}.")
    actual = int(match.group("version"))
    if actual != expected_major:
        raise error_type(
            f"Selected Minecraft target requires Java {expected_major}, but {command!r} "
            f"reports Java {actual}. Configure MMM_JAVA_{expected_major}_CMD to the exact JDK."
        )
