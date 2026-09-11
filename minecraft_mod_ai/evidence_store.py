"""Evidence store for immutable compilation and test evidence records.

P1-3: Real evidence store implementation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompileResult:
    """Result of compilation evidence."""
    status: str  # PASS | FAIL | NOT_RUN
    exit_code: int
    stdout_hash: str
    stderr_hash: str
    duration_ms: int
    compiler_version: str


@dataclass(frozen=True)
class GameTestResult:
    """Result of gametest evidence."""
    status: str  # PASS | FAIL | NOT_RUN
    tests_run: int
    tests_passed: int
    tests_failed: int
    duration_ms: int
    log_hash: str


@dataclass(frozen=True)
class RuntimeTestResult:
    """Result of runtime test evidence."""
    status: str  # PASS | FAIL | NOT_RUN
    test_name: str
    duration_ms: int
    log_hash: str


@dataclass(frozen=True)
class EvidenceRecord:
    """Immutable evidence record for a leaf implementation.
    
    P1-3: This replaces fake evidence_id strings with actual evidence objects.
    """
    evidence_id: str  # SHA-256 of canonical record
    
    # Context
    context_id: str
    leaf_id: str
    implementation_id: str
    implementation_hash: str
    
    # Validators used
    validator_hashes: dict[str, str]  # validator_id -> source_hash
    
    # Type schemas
    input_schema_hashes: dict[str, str]  # port -> schema_hash
    output_schema_hashes: dict[str, str]  # port -> schema_hash
    
    # Environment
    minecraft_version: str
    java_version: str
    fabric_version: str
    loom_version: str
    gradle_version: str
    
    # Test results
    compile_result: CompileResult | None
    gametest_result: GameTestResult | None
    runtime_result: RuntimeTestResult | None
    
    # Artifacts
    logs_hash: str
    artifact_hashes: dict[str, str]  # filename -> hash
    receipt_hash: str
    
    # Metadata
    validated_at: str  # ISO 8601
    validator_agent: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        data = asdict(self)
        # Convert nested dataclasses
        if self.compile_result:
            data["compile_result"] = asdict(self.compile_result)
        if self.gametest_result:
            data["gametest_result"] = asdict(self.gametest_result)
        if self.runtime_result:
            data["runtime_result"] = asdict(self.runtime_result)
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        """Deserialize from dict."""
        # Convert nested dicts to dataclasses
        compile_result = None
        if data.get("compile_result"):
            compile_result = CompileResult(**data["compile_result"])
        
        gametest_result = None
        if data.get("gametest_result"):
            gametest_result = GameTestResult(**data["gametest_result"])
        
        runtime_result = None
        if data.get("runtime_result"):
            runtime_result = RuntimeTestResult(**data["runtime_result"])
        
        return cls(
            evidence_id=data["evidence_id"],
            context_id=data["context_id"],
            leaf_id=data["leaf_id"],
            implementation_id=data["implementation_id"],
            implementation_hash=data["implementation_hash"],
            validator_hashes=data["validator_hashes"],
            input_schema_hashes=data["input_schema_hashes"],
            output_schema_hashes=data["output_schema_hashes"],
            minecraft_version=data["minecraft_version"],
            java_version=data["java_version"],
            fabric_version=data["fabric_version"],
            loom_version=data["loom_version"],
            gradle_version=data["gradle_version"],
            compile_result=compile_result,
            gametest_result=gametest_result,
            runtime_result=runtime_result,
            logs_hash=data["logs_hash"],
            artifact_hashes=data["artifact_hashes"],
            receipt_hash=data["receipt_hash"],
            validated_at=data["validated_at"],
            validator_agent=data["validator_agent"],
        )
    
    @staticmethod
    def compute_evidence_id(record_data: dict[str, Any]) -> str:
        """Compute evidence ID from canonical record representation."""
        # Create canonical JSON (sorted keys, no whitespace)
        canonical = json.dumps(record_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class EvidenceStoreError(Exception):
    """Error in evidence store operations."""
    pass


class EvidenceStore:
    """Store for immutable evidence records.
    
    P1-3: Actual evidence storage, not just string IDs.
    """
    
    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path("data/evidence")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def store_evidence(self, record: EvidenceRecord) -> str:
        """Store immutable evidence record.
        
        Returns:
            evidence_id (SHA-256 of record)
        """
        # Verify evidence_id matches content
        # Must exclude evidence_id itself from hash computation
        record_dict = record.to_dict()
        record_dict_for_hash = {k: v for k, v in record_dict.items() if k != "evidence_id"}
        computed_id = EvidenceRecord.compute_evidence_id(record_dict_for_hash)
        
        if record.evidence_id != computed_id:
            raise EvidenceStoreError(
                f"Evidence ID mismatch: {record.evidence_id[:8]} != {computed_id[:8]}"
            )
        
        # Store as JSON file
        evidence_file = self.storage_path / f"{record.evidence_id}.json"
        
        # Don't overwrite if exists (immutable)
        if evidence_file.exists():
            existing = self.retrieve_evidence(record.evidence_id)
            if existing.to_dict() != record_dict:
                raise EvidenceStoreError(
                    f"Evidence {record.evidence_id[:8]} already exists with different content"
                )
            return record.evidence_id
        
        evidence_file.write_text(json.dumps(record_dict, indent=2), encoding="utf-8")
        return record.evidence_id
    
    def retrieve_evidence(self, evidence_id: str) -> EvidenceRecord:
        """Retrieve evidence by ID.
        
        Raises:
            EvidenceStoreError: If not found
        """
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_id):
            raise EvidenceStoreError("Invalid evidence ID")
        evidence_file = self.storage_path / f"{evidence_id}.json"

        if not evidence_file.exists():
            raise EvidenceStoreError(f"Evidence not found: {evidence_id[:8]}")
        
        data = json.loads(evidence_file.read_text(encoding="utf-8"))
        payload = {k: v for k, v in data.items() if k != "evidence_id"}
        if data.get("evidence_id") != evidence_id or EvidenceRecord.compute_evidence_id(payload) != evidence_id:
            raise EvidenceStoreError("Evidence content hash mismatch")
        return EvidenceRecord.from_dict(data)
    
    def verify_evidence_matches_binding(
        self,
        evidence_id: str,
        expected_implementation_hash: str,
        expected_validator_hashes: dict[str, str],
    ) -> bool:
        """Verify evidence matches expected binding hashes.
        
        P1-3: This ensures evidence is still valid for current implementation.
        """
        try:
            evidence = self.retrieve_evidence(evidence_id)
        except EvidenceStoreError:
            return False
        
        # Check implementation hash
        if evidence.implementation_hash != expected_implementation_hash:
            return False
        
        # Check all validator hashes
        for validator_id, expected_hash in expected_validator_hashes.items():
            if evidence.validator_hashes.get(validator_id) != expected_hash:
                return False
        
        return True
    
    def list_evidence_for_leaf(self, leaf_id: str) -> list[EvidenceRecord]:
        """List all evidence records for a specific leaf."""
        records = []
        for evidence_file in self.storage_path.glob("*.json"):
            try:
                record = self.retrieve_evidence(evidence_file.stem)
                if record.leaf_id == leaf_id:
                    records.append(record)
            except Exception:
                continue
        return records


# Global evidence store
_global_evidence_store: EvidenceStore | None = None


def get_global_evidence_store(storage_path: Path | None = None) -> EvidenceStore:
    """Get global evidence store singleton."""
    global _global_evidence_store
    if _global_evidence_store is None:
        _global_evidence_store = EvidenceStore(storage_path)
    return _global_evidence_store


def reset_global_evidence_store():
    """Reset global evidence store (for testing)."""
    global _global_evidence_store
    _global_evidence_store = None
