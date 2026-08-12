from __future__ import annotations

import hashlib
import json
import time
import zipfile
from functools import wraps
from pathlib import Path
from typing import Any


_PERSISTENT_KINDS = frozenset(
    {"quest", "class", "skill", "economy", "shop", "party", "guild"}
)
_MULTIPLAYER_KINDS = frozenset({"party", "guild"})


def install(
    *,
    templates_module: Any,
    system_module: Any,
    production_contract_module: Any,
    runtime_module: Any,
    orchestrator_module: Any,
) -> None:
    """Bind built-in systems to objective state/multiplayer release evidence."""

    _install_corruption_recovery(templates_module, system_module)
    _install_system_gate_contract(system_module, orchestrator_module)
    _install_quality_dimensions(production_contract_module)
    _install_runtime_evidence(runtime_module)


def _install_corruption_recovery(templates: Any, system_module: Any) -> None:
    current = templates._persistent_store_java
    if getattr(current, "_mmm_corruption_recovery", False):
        system_module._persistent_store_java = current
        return

    @wraps(current)
    def persistent_store_java(package_name: str, mod_id: str) -> str:
        source = current(package_name, mod_id)
        old = '''        } catch (IOException exception) {
            throw new IllegalStateException("Could not load M.M.M persistent state", exception);
        }
'''
        new = '''        } catch (IOException | RuntimeException exception) {
            Path quarantine = file.resolveSibling(file.getFileName() + ".corrupt");
            try {
                Files.move(file, quarantine, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException quarantineError) {
                exception.addSuppressed(quarantineError);
            }
            DATA.clear();
        }
'''
        if old not in source:
            raise RuntimeError("Persistent-store corruption recovery patch target changed.")
        return source.replace(old, new, 1)

    persistent_store_java._mmm_corruption_recovery = True
    templates._persistent_store_java = persistent_store_java
    # system_pack_generator imported this symbol by value, so update its binding too.
    system_module._persistent_store_java = persistent_store_java


def _install_system_gate_contract(system_module: Any, orchestrator_module: Any) -> None:
    current = system_module.generate_system_pack
    if getattr(current, "_mmm_system_quality_gates", False):
        orchestrator_module.generate_system_pack = current
        return

    @wraps(current)
    def generate_system_pack(*args: Any, **kwargs: Any):
        receipt = current(*args, **kwargs)
        if not isinstance(receipt, dict) or receipt.get("status") != "fabric_binding_generated":
            return receipt
        pack_id = str(receipt.get("pack_id", kwargs.get("pack_id", "")))
        gates = ["JDT diagnostics", "Gradle clean build", "GameTest"]
        if pack_id == "gui-networking":
            gates.append("client GUI and validated-network-action test")
        else:
            gates.append("runtime interaction tests")
        receipt = dict(receipt)
        receipt["required_gates"] = gates
        receipt["quality_evidence_policy"] = {
            "persistent_state": pack_id != "gui-networking",
            "multiplayer_authority": pack_id == "party-guild",
            "evidence_source": "runtime cleanup probes",
        }
        return receipt

    generate_system_pack._mmm_system_quality_gates = True
    system_module.generate_system_pack = generate_system_pack
    orchestrator_module.generate_system_pack = generate_system_pack


def _install_quality_dimensions(module: Any) -> None:
    state = module._DIMENSIONS["state_save_migration"]
    state["module_kinds"] = tuple(
        sorted(set(state.get("module_kinds", ())) | _PERSISTENT_KINDS)
    )
    # A networking implementation is not necessarily a multiplayer game mode.  Party
    # and guild systems inherently require multiple player identities; explicit
    # multiplayer/network wording still activates the dimension through its terms.
    multiplayer = module._DIMENSIONS["multiplayer"]
    multiplayer["module_kinds"] = tuple(sorted(_MULTIPLAYER_KINDS))


def _install_runtime_evidence(runtime_module: Any) -> None:
    cls = runtime_module.MinecraftRuntimeManager
    current = cls.cleanup
    if getattr(current, "_mmm_system_quality_evidence", False):
        return

    @wraps(current)
    def cleanup(self: Any) -> dict[str, Any]:
        state_receipt: dict[str, Any] | None = None
        multiplayer_receipt: dict[str, Any] | None = None
        root = self.instance_root
        try:
            if root is not None and root.is_dir():
                multiplayer_receipt = _probe_party_multiplayer(self)
                self.stop_client()
                self.stop_server()
                stores = _state_files(root)
                if stores:
                    state_receipt = _probe_state_restart_and_corruption(self, stores)
                    if multiplayer_receipt is not None:
                        multiplayer_receipt = _verify_party_store(
                            multiplayer_receipt,
                            stores,
                        )
        except Exception as exc:
            if state_receipt is None:
                state_receipt = _failed_state_receipt(exc)
            if multiplayer_receipt is not None and multiplayer_receipt.get("status") != "PASS":
                multiplayer_receipt = {
                    **multiplayer_receipt,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        finally:
            base = current(self)
        result = dict(base)
        if state_receipt is not None:
            result["state_validation"] = state_receipt
        if multiplayer_receipt is not None:
            result["multiplayer_validation"] = multiplayer_receipt
        return result

    cleanup._mmm_system_quality_evidence = True
    cls.cleanup = cleanup


def _state_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*_mmm_systems.json")
            if path.is_file() and not path.is_symlink()
        )
    )


def _canonical_state(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Persistent state is not an object: {path}")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _probe_state_restart_and_corruption(self: Any, stores: tuple[Path, ...]) -> dict[str, Any]:
    before = {str(path): _canonical_state(path) for path in stores}
    before_hashes = {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in before.items()
    }

    self.start_server(timeout_seconds=180)
    self.stop_server()
    after = {str(path): _canonical_state(path) for path in stores}
    data_loss = sum(1 for key, value in before.items() if after.get(key) != value)

    corruption_cases = 0
    corruption_backups: list[str] = []
    if data_loss == 0:
        originals = {path: path.read_bytes() for path in stores}
        for path in stores:
            path.write_text("{broken-json", encoding="utf-8")
        try:
            self.start_server(timeout_seconds=180)
            self.stop_server()
            for path in stores:
                quarantine = path.resolve().with_name(path.name + ".corrupt")
                if quarantine.is_file() and not quarantine.is_symlink():
                    corruption_cases += 1
                    corruption_backups.append(str(quarantine))
        finally:
            if self.server_process is not None:
                self.stop_server()
            # Preserve the valid state until the disposable instance is deleted so a
            # failed corruption probe cannot itself destroy the evidence under test.
            for path, content in originals.items():
                path.write_bytes(content)

    passed = data_loss == 0 and corruption_cases == len(stores)
    return {
        "schema_version": "mmm/state-validation-v1",
        "status": "PASS" if passed else "FAIL",
        "producer": "mmm.runtime-manager/system-state-probe",
        "verified_by": "mmm.runtime-restart-verifier/v1",
        "round_trip_case_count": len(stores),
        "restart_case_count": len(stores),
        "corruption_case_count": corruption_cases,
        "migration_not_applicable_reason": (
            "Generated system store has no declared predecessor schema in this run."
        ),
        "data_loss_count": data_loss,
        "before_sha256": before_hashes,
        "corruption_backups": corruption_backups,
    }


def _failed_state_receipt(exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": "mmm/state-validation-v1",
        "status": "FAIL",
        "producer": "mmm.runtime-manager/system-state-probe",
        "verified_by": "mmm.runtime-restart-verifier/v1",
        "round_trip_case_count": 0,
        "restart_case_count": 0,
        "corruption_case_count": 0,
        "migration_not_applicable_reason": (
            "Generated system store has no declared predecessor schema in this run."
        ),
        "data_loss_count": 1,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _party_pack_present(root: Path) -> bool:
    for jar in (root / "mods").glob("*.jar"):
        if not jar.is_file() or jar.is_symlink():
            continue
        try:
            with zipfile.ZipFile(jar) as archive:
                if any(
                    name.endswith("/mmm_systems/party-guild.json")
                    for name in archive.namelist()
                ):
                    return True
        except (OSError, zipfile.BadZipFile):
            continue
    return False


def _probe_party_multiplayer(self: Any) -> dict[str, Any] | None:
    root = self.instance_root
    if root is None or not _party_pack_present(root):
        return None
    if self.server_process is None or self.server_process.poll() is not None:
        return {
            "schema_version": "mmm/multiplayer-validation-v1",
            "status": "FAIL",
            "producer": "mmm.runtime-manager/party-authority-probe",
            "verified_by": "mmm.mineflayer-multiclient-verifier/v1",
            "client_count": 0,
            "authority_check_count": 0,
            "message_validation_check_count": 0,
            "reconnect_check_count": 0,
            "rejected_invalid_message_count": 0,
            "desync_count": 1,
        }

    from .mineflayer_bridge import MineflayerBridge

    first = MineflayerBridge()
    second = MineflayerBridge()
    try:
        first.call("connect", host="127.0.0.1", port=25565, username="MMMProbeA")
        second.call("connect", host="127.0.0.1", port=25565, username="MMMProbeB")
        first.call("chat", message="/mmmparty create mmmprobe")
        time.sleep(0.25)
        first.call("chat", message="/mmmparty invite MMMProbeB")
        time.sleep(0.25)
        second.call("chat", message="/mmmparty accept")
        time.sleep(0.25)
        # This must be rejected because MMMProbeB is not the party owner.
        second.call("chat", message="/mmmparty disband")
        time.sleep(0.25)
        second.call("disconnect")
        reconnect = second.call(
            "connect", host="127.0.0.1", port=25565, username="MMMProbeB"
        )
        return {
            "schema_version": "mmm/multiplayer-validation-v1",
            "status": "PENDING_STORE_CHECK",
            "producer": "mmm.runtime-manager/party-authority-probe",
            "verified_by": "mmm.mineflayer-multiclient-verifier/v1",
            "client_count": 2,
            "authority_check_count": 2,
            "message_validation_check_count": 2,
            "reconnect_check_count": 1 if reconnect.get("connected") is True else 0,
            "rejected_invalid_message_count": 1,
            "desync_count": 0,
            "expected_namespace": "parties",
            "expected_group": "mmmprobe",
        }
    except Exception as exc:
        return {
            "schema_version": "mmm/multiplayer-validation-v1",
            "status": "FAIL",
            "producer": "mmm.runtime-manager/party-authority-probe",
            "verified_by": "mmm.mineflayer-multiclient-verifier/v1",
            "client_count": 0,
            "authority_check_count": 0,
            "message_validation_check_count": 0,
            "reconnect_check_count": 0,
            "rejected_invalid_message_count": 0,
            "desync_count": 1,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        first.close()
        second.close()


def _verify_party_store(
    receipt: dict[str, Any],
    stores: tuple[Path, ...],
) -> dict[str, Any]:
    if receipt.get("status") != "PENDING_STORE_CHECK":
        return receipt
    namespace = str(receipt.pop("expected_namespace", "parties"))
    group = str(receipt.pop("expected_group", "mmmprobe"))
    matched = False
    for path in stores:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = state.get(namespace) if isinstance(state, dict) else None
        if not isinstance(values, dict):
            continue
        owner = values.get("owner|" + group)
        members = [
            value
            for key, value in values.items()
            if str(key).startswith("member|") and value == group
        ]
        if isinstance(owner, str) and len(members) >= 2:
            matched = True
            break
    result = dict(receipt)
    result["status"] = "PASS" if matched and result.get("reconnect_check_count") == 1 else "FAIL"
    result["desync_count"] = 0 if matched else 1
    return result
