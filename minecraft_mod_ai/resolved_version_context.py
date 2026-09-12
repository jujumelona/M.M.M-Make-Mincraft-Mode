"""Immutable HOST snapshots. Request modes never enter executable contexts."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
from types import MappingProxyType


class VersionContextError(ValueError):
    def __init__(self, code, **details):
        self.diagnostic = {"code": code, **details}
        super().__init__(json.dumps(self.diagnostic, sort_keys=True, ensure_ascii=False))


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@lru_cache(maxsize=256)
def _check_schema_cached(encoded_schema: str):
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(json.loads(encoded_schema))


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class VersionRequest:
    mode: str = "AUTO"
    requested_minecraft: str | None = None
    provenance: str = "DEFAULT_POLICY"

    def __post_init__(self):
        from .target_contract import minecraft_version_tuple

        if self.mode not in {"AUTO", "PINNED"}:
            raise VersionContextError("INVALID_VERSION_REQUEST", mode=self.mode)
        if self.mode == "AUTO":
            if self.requested_minecraft is not None:
                raise VersionContextError("INVALID_VERSION_REQUEST", requested=self.requested_minecraft)
        else:
            if not isinstance(self.requested_minecraft, str):
                raise VersionContextError("INVALID_VERSION_REQUEST", requested=self.requested_minecraft)
            minecraft_version_tuple(self.requested_minecraft)
            if self.requested_minecraft != self.requested_minecraft.strip():
                raise VersionContextError("INVALID_VERSION_REQUEST", requested=self.requested_minecraft)
        if self.provenance not in {"USER", "DEFAULT_POLICY", "HOST"}:
            raise VersionContextError("INVALID_VERSION_PROVENANCE")

    @classmethod
    def from_input(cls, value):
        if value is None or (isinstance(value, str) and value.strip().casefold() in {"", "auto"}):
            return cls()
        return cls("PINNED", value, "USER")


_FACT_FIELDS = frozenset({
    "host_revision", "capabilities", "api_symbols", "schemas", "artifact_rules",
    "dependency_coordinates", "repositories", "replacements", "leaf_bindings",
})


@dataclass(frozen=True)
class ResolvedVersionContext:
    """Canonical JSON storage makes nested facts immutable and serialization explicit."""

    snapshot_json: str

    def __post_init__(self):
        from .target_contract import target_contract_from_mapping

        value = json.loads(self.snapshot_json)
        if set(value) != {"target", "host_facts", "source"} or value["source"] != "HOST":
            raise VersionContextError("INVALID_VERSION_CONTEXT")
        target_contract_from_mapping(value["target"])
        facts = value["host_facts"]
        if not isinstance(facts, dict) or not _FACT_FIELDS <= set(facts) or set(facts) - _FACT_FIELDS - {"api_inspection", "resource_asset_bindings"}:
            raise VersionContextError("HOST_BUNDLE_INCOMPLETE", fields=sorted(_FACT_FIELDS))
        if not isinstance(facts.get("resource_asset_bindings", {}), dict):
            raise VersionContextError("HOST_FACT_TYPE", field="resource_asset_bindings")
        if not isinstance(facts["host_revision"], str) or not facts["host_revision"].strip():
            raise VersionContextError("HOST_REVISION_REQUIRED")
        for key in _FACT_FIELDS - {"host_revision", "repositories"}:
            if not isinstance(facts[key], dict):
                raise VersionContextError("HOST_FACT_TYPE", field=key)
        if any(type(item) is not bool for item in facts["capabilities"].values()):
            raise VersionContextError("HOST_CAPABILITY_TYPE")
        for name, schema in facts["schemas"].items():
            if not isinstance(schema, dict):
                raise VersionContextError("HOST_SCHEMA_INVALID", name=name)
            try:
                _check_schema_cached(_encode(schema))
            except Exception as exc:
                raise VersionContextError("HOST_SCHEMA_INVALID", name=name) from exc
        for name, rule in facts["artifact_rules"].items():
            if not isinstance(rule, dict) or set(rule) != {"template_sha256", "required_symbols", "requires_capabilities"}:
                raise VersionContextError("HOST_ARTIFACT_RULE_INVALID", name=name)
            for category, field in (("api_symbols", "required_symbols"), ("capabilities", "requires_capabilities")):
                if not isinstance(rule[field], list) or any(not isinstance(key, str) or key not in facts[category] for key in rule[field]):
                    raise VersionContextError("HOST_ARTIFACT_RULE_INVALID", name=name, field=field)
        for sym_name, sym_val in facts["api_symbols"].items():
            if isinstance(sym_val, str):
                if not sym_val.strip():
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols")
            elif isinstance(sym_val, dict):
                required_sym_fields = ("owner", "name", "descriptor", "kind", "static", "side", "namespace")
                if not all(k in sym_val for k in required_sym_fields):
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if not isinstance(sym_val["name"], str) or not sym_val["name"].strip():
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if not isinstance(sym_val["owner"], str):
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if not isinstance(sym_val["descriptor"], str):
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if sym_val["kind"] not in {"method", "field", "class", "constructor"}:
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if not isinstance(sym_val["static"], bool):
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if sym_val["side"] not in {"common", "client", "server"}:
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
                if sym_val["namespace"] not in {"minecraft", "fabric"}:
                    raise VersionContextError("HOST_FACT_TYPE", field="api_symbols", symbol=sym_name)
            else:
                raise VersionContextError("HOST_FACT_TYPE", field="api_symbols")
        for key in ("dependency_coordinates", "replacements"):
            if any(not isinstance(item, str) or not item.strip() for item in facts[key].values()):
                raise VersionContextError("HOST_FACT_TYPE", field=key)
        for leaf_name, binding in facts.get("leaf_bindings", {}).items():
            if not isinstance(binding, dict) or "state" not in binding:
                raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_name)
            state = binding["state"]
            if state not in {"admitted", "unsupported", "not_reviewed"}:
                raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_name, state=state)
            if state == "admitted":
                impl = binding.get("implementation")
                if not isinstance(impl, dict):
                    raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_name)
                required_impl_fields = (
                    "implementation_id",
                    "executor_type",
                    "implementation_sha256",
                    "validator_profile",
                    "validator_sha256",
                    "input_schema_sha256",
                    "output_schema_sha256",
                    "evidence_id",
                )
                missing = [f for f in required_impl_fields if not isinstance(impl.get(f), str) or not impl.get(f, "").strip()]
                if missing:
                    raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_name, missing_fields=missing)
                for hash_key in ("implementation_sha256", "validator_sha256", "input_schema_sha256", "output_schema_sha256"):
                    if not impl[hash_key].startswith("sha256:"):
                        raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_name, field=hash_key)
        if not isinstance(facts["repositories"], list) or any(
            not isinstance(url, str) or not url.startswith("https://") for url in facts["repositories"]
        ):
            raise VersionContextError("HOST_REPOSITORIES_INVALID")
        # No floating coordinates may survive the host admission boundary.
        for key in ("minecraft_version", "java_version", "fabric_loader", "fabric_api", "fabric_loom", "gradle"):
            token = str(value["target"][key]).casefold()
            if token in {"auto", "latest", "unresolved", "*"} or "snapshot" in token:
                raise VersionContextError("UNRESOLVED_VERSION_COORDINATE", field=key, actual=token)
        if self.snapshot_json != _encode(value):
            raise VersionContextError("NONCANONICAL_VERSION_CONTEXT")

    @property
    def context_id(self):
        return "sha256:" + sha256(self.snapshot_json.encode()).hexdigest()

    @property
    def minecraft(self):
        return self.to_dict()["target"]["minecraft_version"]

    @property
    def target(self):
        return _freeze(json.loads(self.snapshot_json)["target"])

    @property
    def java(self):
        return int(self.target["java_version"])

    @property
    def host_revision(self):
        return self.facts["host_revision"]

    @property
    def resource_pack_format(self):
        return self.target["resource_pack_format"]

    @property
    def data_pack_version(self) -> str:
        return str(self.target["data_pack_version"])

    @property
    def data_pack_major(self) -> int:
        return int(str(self.target["data_pack_version"]).split(".", 1)[0])

    @property
    def data_pack_format(self) -> str:
        return self.data_pack_version

    @property
    def facts(self):
        return _freeze(json.loads(self.snapshot_json)["host_facts"])

    @property
    def capabilities(self):
        return self.facts["capabilities"]

    @property
    def api_symbols(self):
        return self.facts["api_symbols"]

    @property
    def leaf_bindings(self):
        return self.facts.get("leaf_bindings", {})

    def to_dict(self):
        return {**json.loads(self.snapshot_json), "context_id": self.context_id}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise VersionContextError("INVALID_VERSION_CONTEXT")
        raw = dict(value)
        expected = raw.pop("context_id", None)
        result = cls(_encode(raw))
        if expected != result.context_id:
            raise VersionContextError("VERSION_CONTEXT_HASH_MISMATCH", expected=expected, actual=result.context_id)
        return result

    @classmethod
    def from_target(cls, target):
        target.validate()
        raw = asdict(target)
        facts = json.loads(raw.pop("host_facts_json", "") or "{}")
        raw["deterministic_module_kinds"] = sorted(raw["deterministic_module_kinds"])
        return cls(_encode({"target": raw, "host_facts": facts, "source": "HOST"}))

    def require_fact(self, category, name):
        values = self.facts.get(category)
        if not isinstance(values, Mapping) or name not in values:
            raise VersionContextError("HOST_FACT_UNAVAILABLE", category=category, name=name, context_id=self.context_id)
        return values[name]

    def require_capability(self, name):
        if self.require_fact("capabilities", name) is not True:
            raise VersionContextError("UNSUPPORTED_CAPABILITY", capability=name, context_id=self.context_id)

    def assert_context(self, context_id, *, code="VERSION_CONTEXT_MISMATCH"):
        if context_id != self.context_id:
            raise VersionContextError(code, expected=self.context_id, actual=context_id)

    def admit_template(self, template):
        rule = self.require_fact("artifact_rules", template["id"])
        actual = "sha256:" + sha256(_encode(template).encode()).hexdigest()
        if rule.get("template_sha256") != actual:
            raise VersionContextError("HOST_TEMPLATE_NOT_ADMITTED", artifact=template["id"], context_id=self.context_id)
        for capability in rule.get("requires_capabilities", ()):
            self.require_capability(capability)
        for name in rule.get("required_symbols", ()):
            self.require_fact("api_symbols", name)
        return rule

    def require_leaf_binding(self, leaf_id: str):
        bindings = self.facts.get("leaf_bindings", {})
        if not isinstance(bindings, Mapping) or leaf_id not in bindings:
            raise VersionContextError("HOST_FACT_UNAVAILABLE", category="leaf_bindings", name=leaf_id, context_id=self.context_id)
        binding = bindings[leaf_id]
        state = binding.get("state") if isinstance(binding, Mapping) else None
        if state != "admitted":
            raise VersionContextError("UNSUPPORTED_LEAF", leaf=leaf_id, state=state, context_id=self.context_id)
        impl = binding.get("implementation")
        if not isinstance(impl, Mapping):
            raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_id)
        required_impl_fields = (
            "implementation_id",
            "executor_type",
            "implementation_sha256",
            "validator_profile",
            "validator_sha256",
            "input_schema_sha256",
            "output_schema_sha256",
            "evidence_id",
        )
        missing = [f for f in required_impl_fields if not isinstance(impl.get(f), str) or not impl.get(f, "").strip()]
        if missing:
            raise VersionContextError("HOST_LEAF_BINDING_INVALID", leaf=leaf_id, missing_fields=missing)
        return binding

    def validate_artifact(self, template, output):
        from jsonschema import Draft202012Validator
        import re

        rule = self.admit_template(template)
        template_id = template.get("id", "")
        lang = template.get("render", {}).get("language")

        # 0. Strip comments and string literals for AST/executable code checks
        stripped = re.sub(
            r'(/\*[\s\S]*?\*/)|(//[^\r\n]*)|("(?:\\.|[^"\\])*")|(\'(?:\\.|[^\'\\])*\')',
            " ",
            output,
        )

        # 1. Symbol and invocation structure validation
        for name in rule.get("required_symbols", ()):
            sym = self.require_fact("api_symbols", name)
            if isinstance(sym, Mapping):
                owner = sym.get("owner", "")
                sym_name = sym.get("name", "")
                kind = sym.get("kind", "method")
                is_static = bool(sym.get("static", True))
                short_owner = owner.split(".")[-1] if owner else ""
                owner_candidates = [c for c in (owner, short_owner) if c]
                if short_owner == "ResourceKey":
                    owner_candidates.append("RegistryKey")
                elif short_owner == "RegistryKey":
                    owner_candidates.append("ResourceKey")

                owner_regex = "|".join(re.escape(c) for c in owner_candidates)
                if kind == "method":
                    if is_static:
                        pattern = rf"\b(?:{owner_regex})\s*\.\s*{re.escape(sym_name)}\s*\("
                    else:
                        pattern = rf"\.\s*{re.escape(sym_name)}\s*\("
                    if not re.search(pattern, stripped):
                        raise VersionContextError(
                            "INVALID_API_INVOCATION",
                            artifact=template_id,
                            symbol=name,
                            expected_invocation=f"{owner}.{sym_name}",
                            context_id=self.context_id,
                            repair_scope=[template_id],
                        )
                elif kind == "field":
                    pattern = rf"\b(?:{owner_regex})\s*\.\s*{re.escape(sym_name)}\b"
                    if not re.search(pattern, stripped):
                        raise VersionContextError(
                            "INVALID_API_SYMBOL",
                            artifact=template_id,
                            symbol=name,
                            expected_symbol=f"{owner}.{sym_name}",
                            context_id=self.context_id,
                            repair_scope=[template_id],
                        )
                elif kind == "class":
                    pattern = rf"\b(?:{owner_regex})\b"
                    if not re.search(pattern, stripped):
                        raise VersionContextError(
                            "INVALID_API_SYMBOL",
                            artifact=template_id,
                            symbol=name,
                            expected_symbol=owner,
                            context_id=self.context_id,
                            repair_scope=[template_id],
                        )
                elif kind == "constructor":
                    pattern = rf"\bnew\s+(?:{owner_regex})\s*\("
                    if not re.search(pattern, stripped):
                        raise VersionContextError(
                            "INVALID_API_INVOCATION",
                            artifact=template_id,
                            symbol=name,
                            expected_invocation=f"new {owner}",
                            context_id=self.context_id,
                            repair_scope=[template_id],
                        )
            else:
                expected = str(sym)
                if expected not in output:
                    raise VersionContextError(
                        "INVALID_API_SYMBOL",
                        artifact=template_id,
                        symbol=name,
                        expected_symbol=expected,
                        context_id=self.context_id,
                        repair_scope=[template_id],
                    )
                if "." in expected and any(expected.endswith(suffix) for suffix in (".register", ".create", ".of", ".stacksTo")):
                    escaped = re.escape(expected)
                    if not re.search(rf"\b{escaped}\s*\(", output) and not re.search(rf"\.\s*{re.escape(expected.lstrip('.'))}\s*\(", output):
                        raise VersionContextError(
                            "INVALID_API_INVOCATION",
                            artifact=template_id,
                            symbol=name,
                            expected_invocation=expected,
                            context_id=self.context_id,
                            repair_scope=[template_id],
                        )

        # 2. Side constraint validation (common/server must not reference client-only classes)
        template_side = template.get("side", ("common",))
        if isinstance(template_side, str):
            template_side = (template_side,)
        if "client" not in template_side:
            if "net.minecraft.client." in stripped or "net/minecraft/client/" in stripped:
                raise VersionContextError(
                    "CROSS_SIDE_LEAKAGE",
                    artifact=template_id,
                    side=list(template_side),
                    context_id=self.context_id,
                    repair_scope=[template_id],
                )

        # 3. JSON schema validation
        if lang == "json":
            schema = self.to_dict()["host_facts"]["schemas"].get(template_id)
            if schema is None:
                raise VersionContextError("HOST_FACT_UNAVAILABLE", category="schemas", name=template_id)
            Draft202012Validator(schema).validate(json.loads(output))

        return {
            "type": "HOST_TEMPLATE_CONTRACT",
            "status": "PASS",
            "context_id": self.context_id,
            "artifact": template_id,
            "repair_scope": [template_id],
        }


class VersionResolver:
    """Select a whole admitted bundle; never independently select dependencies."""

    def __init__(self, bundles, *, auto_context_id):
        self._bundles = tuple(bundles)
        if not all(isinstance(bundle, ResolvedVersionContext) for bundle in self._bundles):
            raise VersionContextError("INVALID_HOST_BUNDLE")
        versions = [bundle.minecraft for bundle in self._bundles]
        if len(versions) != len(set(versions)):
            raise VersionContextError("AMBIGUOUS_HOST_BUNDLE", supported=versions)
        if sum(bundle.context_id == auto_context_id for bundle in self._bundles) != 1:
            raise VersionContextError("HOST_AUTO_BUNDLE_REQUIRED")
        self._auto_context_id = auto_context_id

    def resolve(self, request: VersionRequest):
        if request.mode == "AUTO":
            matches = [bundle for bundle in self._bundles if bundle.context_id == self._auto_context_id]
        else:
            matches = [bundle for bundle in self._bundles if bundle.minecraft == request.requested_minecraft]
        if len(matches) != 1:
            raise VersionContextError(
                "UNSUPPORTED_MINECRAFT_VERSION" if not matches else "AMBIGUOUS_HOST_BUNDLE",
                requested=request.requested_minecraft,
                supported=sorted({bundle.minecraft for bundle in self._bundles}),
            )
        return matches[0]


def execution_context(context, job):
    """Reject missing or mixed snapshots before generation, reuse, or materialization."""
    raw = (context or {}).get("resolved_version_context")
    identifier = getattr(job, "context_id", "")
    if raw is None:
        if identifier or getattr(job, "canonical_leaf", "") or getattr(job, "implementation_id", ""):
            raise VersionContextError("VERSION_CONTEXT_REQUIRED", artifact=job.job_id)
        return None
    if isinstance(raw, ResolvedVersionContext):
        resolved = raw
    else:
        resolved = ResolvedVersionContext.from_dict(raw)
    resolved.assert_context(identifier)
    actual = job.deterministic_inputs.get("minecraft_version")
    if actual is not None and actual != resolved.minecraft:
        raise VersionContextError("HOST_FACT_OVERRIDE", field="minecraft_version", actual=actual)
    return resolved
