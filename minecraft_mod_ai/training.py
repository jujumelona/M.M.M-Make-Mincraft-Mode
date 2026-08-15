from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .config_paths import config_path


class TrainingTraceError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedTrainingTrace:
    trace_id: str
    task: str
    prompt: str
    response: str
    patch: str
    minecraft_version: str
    loader: str
    java_version: int
    mappings: str
    fabric_api: str
    source_license: str
    source_commit: str
    gradle_exit_code: int
    diagnostics_error_count: int
    request_fidelity_passed: bool
    gametest_passed: bool
    jar_validation_passed: bool
    registry_references_valid: bool
    requested_feature_deleted: bool = False
    cross_loader_api: bool = False
    wrong_version_symbol: bool = False
    approval_scope_escape: bool = False

    def validate(self, policy: dict[str, Any]) -> None:
        target = policy["target"]
        if target.get("mode") == "trace_exact":
            exact_fields = {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "java_version": str(self.java_version),
                "mappings": self.mappings,
                "fabric_api": self.fabric_api,
            }
            missing = [key for key, value in exact_fields.items() if not str(value).strip()]
            if missing:
                raise TrainingTraceError(
                    "Training trace target is incomplete: " + ", ".join(missing)
                )
        else:
            if self.minecraft_version != target["minecraft_version"]:
                raise TrainingTraceError("Wrong Minecraft version.")
            if self.loader != target["loader"]:
                raise TrainingTraceError("Wrong mod loader.")
            if self.java_version != int(target["java_version"]):
                raise TrainingTraceError("Wrong Java version.")
            if self.mappings != target["mappings"]:
                raise TrainingTraceError("Wrong mappings.")
            if self.fabric_api != target["fabric_api"]:
                raise TrainingTraceError("Wrong Fabric API.")
        if self.source_license not in set(policy["allowed_source_licenses"]):
            raise TrainingTraceError(
                f"Source license is not allowlisted: {self.source_license}"
            )
        if not self.source_commit.strip():
            raise TrainingTraceError("source_commit is required.")
        if self.gradle_exit_code != 0:
            raise TrainingTraceError("Only successful Gradle traces may enter SFT.")
        if self.diagnostics_error_count != 0:
            raise TrainingTraceError("Java diagnostics contain errors.")
        if not self.request_fidelity_passed:
            raise TrainingTraceError("Request fidelity did not pass.")
        if not self.gametest_passed:
            raise TrainingTraceError("GameTest did not pass.")
        if not self.jar_validation_passed:
            raise TrainingTraceError("JAR validation did not pass.")
        if not self.registry_references_valid:
            raise TrainingTraceError("Registry references are invalid.")
        if any(
            (
                self.requested_feature_deleted,
                self.cross_loader_api,
                self.wrong_version_symbol,
                self.approval_scope_escape,
            )
        ):
            raise TrainingTraceError("Trace contains a disqualifying shortcut or scope violation.")

    def reward(self, policy: dict[str, Any]) -> float:
        values = policy["reward"]
        total = 0.0
        total += float(values["gradle_build"]) if self.gradle_exit_code == 0 else 0.0
        total += (
            float(values["diagnostics_clean"])
            if self.diagnostics_error_count == 0
            else 0.0
        )
        total += (
            float(values["registry_references_valid"])
            if self.registry_references_valid
            else 0.0
        )
        total += float(values["gametest"]) if self.gametest_passed else 0.0
        total += (
            float(values["jar_validation"])
            if self.jar_validation_passed
            else 0.0
        )
        total += (
            float(values["request_fidelity"])
            if self.request_fidelity_passed
            else 0.0
        )
        if self.requested_feature_deleted:
            total += float(values["requested_feature_deleted"])
        if self.cross_loader_api:
            total += float(values["cross_loader_api"])
        if self.wrong_version_symbol:
            total += float(values["wrong_version_symbol"])
        if self.approval_scope_escape:
            total += float(values["approval_scope_escape"])
        return total


class TrainingTraceStore:
    schema_version = "mmm/training-trace-store-v1"

    def __init__(
        self,
        root: str | Path,
        *,
        policy_path: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        path = (
            Path(policy_path).expanduser().resolve()
            if policy_path is not None
            else config_path("training_policy.yaml")
        )
        self.policy = yaml.safe_load(path.read_text(encoding="utf-8"))
        if self.policy.get("schema_version") != "mmm/training-policy-v1":
            raise TrainingTraceError("Unsupported training policy.")

    def record(self, raw: dict[str, Any]) -> dict[str, Any]:
        trace = _parse_trace(raw)
        trace.validate(self.policy)
        identity_payload = asdict(trace)
        identity_payload["trace_id"] = ""
        identity_json = json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        expected_id = "sha256:" + hashlib.sha256(
            identity_json.encode("utf-8")
        ).hexdigest()
        if trace.trace_id != expected_id:
            trace = VerifiedTrainingTrace(**{**asdict(trace), "trace_id": expected_id})
        canonical = json.dumps(asdict(trace), ensure_ascii=False, sort_keys=True)
        path = self.root / f"{trace.trace_id.removeprefix('sha256:')}.json"
        if path.exists():
            if path.read_text(encoding="utf-8") != canonical:
                raise TrainingTraceError("Trace hash collision or modified duplicate.")
        else:
            path.write_text(canonical, encoding="utf-8")
        return {
            "schema_version": "mmm/training-trace-record-v1",
            "trace_id": trace.trace_id,
            "path": str(path),
            "reward": trace.reward(self.policy),
        }

    def export_sft(self, output_path: str | Path) -> dict[str, Any]:
        traces = list(self.iter_traces())
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for trace in traces:
                trace.validate(self.policy)
                record = {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                f"You are the MinecraftCoder for Minecraft {trace.minecraft_version} "
                                f"{trace.loader}, mappings {trace.mappings}, Java {trace.java_version}. "
                                "Produce only exact-target, request-faithful code or a minimal patch."
                            ),
                        },
                        {"role": "user", "content": trace.prompt},
                        {"role": "assistant", "content": trace.response},
                    ],
                    "metadata": {
                        "trace_id": trace.trace_id,
                        "task": trace.task,
                        "minecraft_version": trace.minecraft_version,
                        "loader": trace.loader,
                        "mappings": trace.mappings,
                        "source_license": trace.source_license,
                        "source_commit": trace.source_commit,
                        "reward": trace.reward(self.policy),
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "schema_version": "mmm/training-export-v1",
            "output_path": str(output),
            "records": len(traces),
            "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
        }

    def iter_traces(self) -> Iterable[VerifiedTrainingTrace]:
        for path in sorted(self.root.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            trace = _parse_trace(raw)
            trace.validate(self.policy)
            yield trace


def make_trace_id(raw: dict[str, Any]) -> str:
    payload = dict(raw)
    payload["trace_id"] = ""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _parse_trace(raw: dict[str, Any]) -> VerifiedTrainingTrace:
    fields = set(VerifiedTrainingTrace.__dataclass_fields__)
    unknown = set(raw) - fields
    if unknown:
        raise TrainingTraceError(f"Unknown trace fields: {sorted(unknown)}")
    try:
        return VerifiedTrainingTrace(**raw)
    except TypeError as exc:
        raise TrainingTraceError(str(exc)) from exc
