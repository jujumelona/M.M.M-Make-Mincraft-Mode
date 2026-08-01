from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .complete_spec import ProductionModule
from .project_edit import inspect_fabric_project, write_text_files
from .scale_policy import ScalePolicy
from .source_patch import sha256_bytes


class LocalAiSidecarGenerationError(RuntimeError):
    pass


INTEGRATION_TYPE = "mmm_local_ai_sidecar"
TOKEN_SYSTEM_PROPERTY = "mmm.sidecar.token"
TOKEN_ENVIRONMENT_VARIABLE = "MMM_SIDECAR_TOKEN"
ALLOWED_CAPABILITIES = frozenset(
    {
        "agent_tool_use",
        "ai_inference",
        "speech_recognition",
        "speech_synthesis",
        "translation",
        "voice_activity_detection",
        "voice_adaptation",
        "voice_conversion",
    }
)
_ALLOWED_CONFIG_FIELDS = frozenset(
    {
        "integration_type",
        "port",
        "timeout_ms",
        "max_request_bytes",
        "max_response_bytes",
        "max_in_flight",
        "capabilities",
        "authentication",
    }
)
_PACKAGE = re.compile(r"^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*$")
_MODULE_ID = re.compile(r"^[a-z][a-z0-9_]{1,127}$")


@dataclass(frozen=True)
class LocalAiSidecarPolicy:
    port: int
    timeout_ms: int
    max_request_bytes: int
    max_response_bytes: int
    max_in_flight: int
    capabilities: tuple[str, ...]
    authentication: str

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/mmm/infer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "integration_type": INTEGRATION_TYPE,
            "port": self.port,
            "timeout_ms": self.timeout_ms,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_in_flight": self.max_in_flight,
            "capabilities": list(self.capabilities),
            "authentication": self.authentication,
        }


def normalize_local_ai_sidecar_config(
    config: Mapping[str, Any],
) -> LocalAiSidecarPolicy:
    if not isinstance(config, Mapping):
        raise LocalAiSidecarGenerationError("Local AI sidecar config must be an object.")
    unknown = set(config) - _ALLOWED_CONFIG_FIELDS
    if unknown:
        raise LocalAiSidecarGenerationError(
            "Unknown local AI sidecar config fields: " + ", ".join(sorted(unknown))
        )
    if config.get("integration_type") != INTEGRATION_TYPE:
        raise LocalAiSidecarGenerationError(
            f"integration_type must equal {INTEGRATION_TYPE!r}."
        )

    port = _bounded_int(config, "port", default=8765, minimum=1024, maximum=65535)
    timeout_ms = _bounded_int(
        config,
        "timeout_ms",
        default=5000,
        minimum=100,
        maximum=30000,
    )
    max_request_bytes = _bounded_int(
        config,
        "max_request_bytes",
        default=262144,
        minimum=256,
        maximum=1048576,
    )
    max_response_bytes = _bounded_int(
        config,
        "max_response_bytes",
        default=262144,
        minimum=256,
        maximum=1048576,
    )
    max_in_flight = _bounded_int(
        config,
        "max_in_flight",
        default=4,
        minimum=1,
        maximum=32,
    )

    raw_capabilities = config.get("capabilities")
    if type(raw_capabilities) is not list or not raw_capabilities:
        raise LocalAiSidecarGenerationError(
            "capabilities must be a non-empty JSON array."
        )
    if any(not isinstance(value, str) for value in raw_capabilities):
        raise LocalAiSidecarGenerationError("Every capability must be a string.")
    capabilities = tuple(sorted(set(raw_capabilities)))
    if len(capabilities) != len(raw_capabilities):
        raise LocalAiSidecarGenerationError("capabilities must not contain duplicates.")
    unsupported = set(capabilities) - ALLOWED_CAPABILITIES
    if unsupported:
        raise LocalAiSidecarGenerationError(
            "Unsupported local AI sidecar capabilities: "
            + ", ".join(sorted(unsupported))
        )

    authentication = config.get("authentication", "none")
    if authentication not in {"none", "external_token"}:
        raise LocalAiSidecarGenerationError(
            "authentication must be 'none' or 'external_token'."
        )
    return LocalAiSidecarPolicy(
        port=port,
        timeout_ms=timeout_ms,
        max_request_bytes=max_request_bytes,
        max_response_bytes=max_response_bytes,
        max_in_flight=max_in_flight,
        capabilities=capabilities,
        authentication=str(authentication),
    )


def local_ai_sidecar_source_path(package_name: str, module_id: str) -> str:
    class_name = local_ai_sidecar_class_name(module_id)
    return (
        "src/main/java/"
        + package_name.replace(".", "/")
        + f"/integration/{class_name}.java"
    )


def local_ai_sidecar_manifest_path(module_id: str) -> str:
    return f".minecraft_ai/integrations/{module_id}.local-ai-sidecar.json"


def local_ai_sidecar_class_name(module_id: str) -> str:
    if not _MODULE_ID.fullmatch(module_id):
        raise LocalAiSidecarGenerationError(f"Invalid sidecar module id: {module_id!r}")
    return "".join(part.capitalize() for part in module_id.split("_")) + "LocalAiSidecar"


def render_local_ai_sidecar_source(
    *,
    package_name: str,
    module_id: str,
    config: Mapping[str, Any] | LocalAiSidecarPolicy,
) -> str:
    if not _PACKAGE.fullmatch(package_name):
        raise LocalAiSidecarGenerationError(f"Invalid Java package: {package_name!r}")
    policy = (
        config
        if isinstance(config, LocalAiSidecarPolicy)
        else normalize_local_ai_sidecar_config(config)
    )
    class_name = local_ai_sidecar_class_name(module_id)
    capabilities = ",\n            ".join(
        f"Capability.{value.upper()}" for value in policy.capabilities
    )
    enum_values = ",\n        ".join(
        f'{value.upper()}("{value}")' for value in sorted(ALLOWED_CAPABILITIES)
    )
    auth_required = "true" if policy.authentication == "external_token" else "false"
    return f'''package {package_name}.integration;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Flow;
import java.util.concurrent.Semaphore;

/**
 * Generated, fail-closed boundary for one reviewed localhost AI sidecar.
 * This utility returns typed data only and never mutates Minecraft world state.
 */
public final class {class_name} {{
    private static final URI ENDPOINT = URI.create("{policy.endpoint}");
    private static final int TIMEOUT_MS = {policy.timeout_ms};
    private static final int MAX_REQUEST_BYTES = {policy.max_request_bytes};
    private static final int MAX_RESPONSE_BYTES = {policy.max_response_bytes};
    private static final boolean AUTHENTICATION_REQUIRED = {auth_required};
    private static final Set<Capability> ENABLED_CAPABILITIES = Set.of(
            {capabilities}
    );
    private static final Semaphore IN_FLIGHT = new Semaphore({policy.max_in_flight});
    private static final HttpClient CLIENT = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(TIMEOUT_MS))
            .followRedirects(HttpClient.Redirect.NEVER)
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    private {class_name}() {{
    }}

    public static CompletableFuture<InferenceResponse> infer(
            String requestId,
            Capability capability,
            JsonObject input
    ) {{
        if (!isValidRequestId(requestId)) {{
            return failed("requestId must use 1-128 safe identifier characters");
        }}
        if (capability == null || !ENABLED_CAPABILITIES.contains(capability)) {{
            return failed("capability is not enabled by the approved policy");
        }}

        JsonObject request = new JsonObject();
        request.addProperty("schema_version", "mmm/sidecar-inference-request-v1");
        request.addProperty("request_id", requestId);
        request.addProperty("capability", capability.wireName());
        request.add("input", input == null ? new JsonObject() : input.deepCopy());
        byte[] requestBytes = request.toString().getBytes(StandardCharsets.UTF_8);
        if (requestBytes.length > MAX_REQUEST_BYTES) {{
            return failed("request exceeds the approved byte limit");
        }}

        String token;
        try {{
            token = externalToken();
        }} catch (SidecarException error) {{
            return CompletableFuture.failedFuture(error);
        }}
        if (!IN_FLIGHT.tryAcquire()) {{
            return failed("sidecar concurrency limit reached");
        }}

        HttpRequest.Builder builder = HttpRequest.newBuilder(ENDPOINT)
                .timeout(Duration.ofMillis(TIMEOUT_MS))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json");
        if (token != null) {{
            builder.header("Authorization", "Bearer " + token);
        }}
        HttpRequest httpRequest;
        try {{
            httpRequest = builder.POST(HttpRequest.BodyPublishers.ofByteArray(requestBytes)).build();
        }} catch (RuntimeException error) {{
            IN_FLIGHT.release();
            return CompletableFuture.failedFuture(error);
        }}

        try {{
            return CLIENT.sendAsync(httpRequest, {class_name}::boundedBody)
                    .thenApplyAsync(response -> parseResponse(requestId, response))
                    .whenComplete((unused, error) -> IN_FLIGHT.release());
        }} catch (RuntimeException error) {{
            IN_FLIGHT.release();
            return CompletableFuture.failedFuture(error);
        }}
    }}

    private static InferenceResponse parseResponse(
            String expectedRequestId,
            HttpResponse<byte[]> response
    ) {{
        try {{
            if (response.statusCode() != 200) {{
                throw new SidecarException("sidecar returned HTTP " + response.statusCode());
            }}
            byte[] bytes = response.body();
            JsonElement parsed = JsonParser.parseString(new String(bytes, StandardCharsets.UTF_8));
            if (!parsed.isJsonObject()) {{
                throw new SidecarException("sidecar response must be a JSON object");
            }}
            return typedResponse(expectedRequestId, parsed.getAsJsonObject());
        }} catch (RuntimeException error) {{
            if (error instanceof SidecarException sidecarError) {{
                throw sidecarError;
            }}
            throw new SidecarException("sidecar response could not be decoded", error);
        }}
    }}

    private static HttpResponse.BodySubscriber<byte[]> boundedBody(
            HttpResponse.ResponseInfo responseInfo
    ) {{
        long declaredLength = responseInfo.headers()
                .firstValueAsLong("Content-Length")
                .orElse(-1L);
        return new BoundedBodySubscriber(MAX_RESPONSE_BYTES, declaredLength);
    }}

    private static InferenceResponse typedResponse(
            String expectedRequestId,
            JsonObject response
    ) {{
        Set<String> allowedFields = Set.of(
                "schema_version", "request_id", "status", "output", "error"
        );
        if (!allowedFields.containsAll(response.keySet())) {{
            throw new SidecarException("sidecar response contains unknown fields");
        }}
        String schemaVersion = requiredString(response, "schema_version", 64);
        String requestId = requiredString(response, "request_id", 128);
        String status = requiredString(response, "status", 16);
        if (!"mmm/sidecar-inference-response-v1".equals(schemaVersion)) {{
            throw new SidecarException("sidecar response schema is unsupported");
        }}
        if (!expectedRequestId.equals(requestId)) {{
            throw new SidecarException("sidecar response request_id mismatch");
        }}
        if (!status.equals("ok") && !status.equals("error")) {{
            throw new SidecarException("sidecar response status is invalid");
        }}
        JsonObject output = optionalObject(response, "output");
        JsonObject error = optionalObject(response, "error");
        String errorCode = optionalString(error, "code", 128);
        String errorMessage = optionalString(error, "message", 1024);
        if (status.equals("ok") && error != null) {{
            throw new SidecarException("successful sidecar response may not contain error");
        }}
        if (status.equals("error") && (errorCode == null || errorMessage == null)) {{
            throw new SidecarException("error response requires typed code and message");
        }}
        return new InferenceResponse(
                requestId,
                status,
                output == null ? new JsonObject() : output,
                errorCode,
                errorMessage
        );
    }}

    private static String requiredString(JsonObject object, String key, int maxLength) {{
        String value = optionalString(object, key, maxLength);
        if (value == null || value.isBlank()) {{
            throw new SidecarException("sidecar response is missing " + key);
        }}
        return value;
    }}

    private static String optionalString(JsonObject object, String key, int maxLength) {{
        if (object == null || !object.has(key) || object.get(key).isJsonNull()) {{
            return null;
        }}
        JsonElement value = object.get(key);
        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {{
            throw new SidecarException(key + " must be a string");
        }}
        String text = value.getAsString();
        if (text.length() > maxLength) {{
            throw new SidecarException(key + " exceeds its character limit");
        }}
        return text;
    }}

    private static JsonObject optionalObject(JsonObject object, String key) {{
        if (!object.has(key) || object.get(key).isJsonNull()) {{
            return null;
        }}
        if (!object.get(key).isJsonObject()) {{
            throw new SidecarException(key + " must be an object");
        }}
        return object.getAsJsonObject(key).deepCopy();
    }}

    private static boolean isValidRequestId(String value) {{
        if (value == null || value.isEmpty() || value.length() > 128) {{
            return false;
        }}
        for (int index = 0; index < value.length(); index++) {{
            char character = value.charAt(index);
            boolean safe = Character.isLetterOrDigit(character)
                    || character == '.' || character == '_' || character == ':' || character == '-';
            if (!safe) {{
                return false;
            }}
        }}
        return true;
    }}

    private static String externalToken() {{
        if (!AUTHENTICATION_REQUIRED) {{
            return null;
        }}
        String token = System.getProperty("{TOKEN_SYSTEM_PROPERTY}");
        if (token == null || token.isBlank()) {{
            token = System.getenv("{TOKEN_ENVIRONMENT_VARIABLE}");
        }}
        if (token == null || token.isBlank()) {{
            throw new SidecarException("external sidecar token is required");
        }}
        if (token.length() > 512 || token.indexOf('\\r') >= 0 || token.indexOf('\\n') >= 0) {{
            throw new SidecarException("external sidecar token is invalid");
        }}
        return token;
    }}

    private static <T> CompletableFuture<T> failed(String message) {{
        return CompletableFuture.failedFuture(new SidecarException(message));
    }}

    private static final class BoundedBodySubscriber
            implements HttpResponse.BodySubscriber<byte[]> {{
        private final int limit;
        private final ByteArrayOutputStream buffer;
        private final CompletableFuture<byte[]> body = new CompletableFuture<>();
        private final boolean declaredTooLarge;
        private Flow.Subscription subscription;

        private BoundedBodySubscriber(int limit, long declaredLength) {{
            this.limit = limit;
            this.buffer = new ByteArrayOutputStream(Math.min(limit, 8192));
            this.declaredTooLarge = declaredLength > limit;
            if (declaredTooLarge) {{
                body.completeExceptionally(
                        new SidecarException("response exceeds the approved byte limit")
                );
            }}
        }}

        @Override
        public CompletionStage<byte[]> getBody() {{
            return body;
        }}

        @Override
        public void onSubscribe(Flow.Subscription value) {{
            if (subscription != null) {{
                value.cancel();
                return;
            }}
            subscription = value;
            if (declaredTooLarge) {{
                value.cancel();
            }} else {{
                value.request(1);
            }}
        }}

        @Override
        public void onNext(List<ByteBuffer> chunks) {{
            if (body.isDone()) {{
                subscription.cancel();
                return;
            }}
            for (ByteBuffer chunk : chunks) {{
                int length = chunk.remaining();
                if (length > limit - buffer.size()) {{
                    subscription.cancel();
                    body.completeExceptionally(
                            new SidecarException("response exceeds the approved byte limit")
                    );
                    return;
                }}
                byte[] bytes = new byte[length];
                chunk.get(bytes);
                buffer.write(bytes, 0, bytes.length);
            }}
            subscription.request(1);
        }}

        @Override
        public void onError(Throwable error) {{
            body.completeExceptionally(error);
        }}

        @Override
        public void onComplete() {{
            body.complete(buffer.toByteArray());
        }}
    }}

    public enum Capability {{
        {enum_values};

        private final String wireName;

        Capability(String wireName) {{
            this.wireName = wireName;
        }}

        public String wireName() {{
            return wireName;
        }}
    }}

    public record InferenceResponse(
            String requestId,
            String status,
            JsonObject output,
            String errorCode,
            String errorMessage
    ) {{
        public InferenceResponse {{
            output = output.deepCopy();
        }}

        @Override
        public JsonObject output() {{
            return output.deepCopy();
        }}
    }}

    public static final class SidecarException extends RuntimeException {{
        public SidecarException(String message) {{
            super(message);
        }}

        public SidecarException(String message, Throwable cause) {{
            super(message, cause);
        }}
    }}
}}
'''


def build_local_ai_sidecar_manifest(
    *,
    package_name: str,
    module_id: str,
    config: Mapping[str, Any] | LocalAiSidecarPolicy,
) -> dict[str, Any]:
    policy = (
        config
        if isinstance(config, LocalAiSidecarPolicy)
        else normalize_local_ai_sidecar_config(config)
    )
    source_path = local_ai_sidecar_source_path(package_name, module_id)
    source = render_local_ai_sidecar_source(
        package_name=package_name,
        module_id=module_id,
        config=policy,
    )
    return {
        "schema_version": "mmm/local-ai-sidecar-policy-v1",
        "module_id": module_id,
        "source": {
            "path": source_path,
            "sha256": sha256_bytes(source.encode("utf-8")),
            "validation": "exact_reconstruction_required",
        },
        "network": {
            "endpoint": policy.endpoint,
            "method": "POST",
            "redirects": "disabled",
            "transport": "java_17_httpclient_send_async",
            "timeout_ms": policy.timeout_ms,
            "max_request_bytes": policy.max_request_bytes,
            "max_response_bytes": policy.max_response_bytes,
            "max_in_flight": policy.max_in_flight,
        },
        "capabilities": list(policy.capabilities),
        "authority": {
            "returns_typed_json_only": True,
            "minecraft_world_mutation": "none",
            "synchronous_wait": "forbidden",
        },
        "secrets": {
            "embedded": False,
            "authentication": policy.authentication,
            "system_property": (
                TOKEN_SYSTEM_PROPERTY
                if policy.authentication == "external_token"
                else None
            ),
            "environment_variable": (
                TOKEN_ENVIRONMENT_VARIABLE
                if policy.authentication == "external_token"
                else None
            ),
        },
        "approved_config": policy.to_dict(),
    }


def render_local_ai_sidecar_manifest(
    *,
    package_name: str,
    module_id: str,
    config: Mapping[str, Any] | LocalAiSidecarPolicy,
) -> str:
    return json.dumps(
        build_local_ai_sidecar_manifest(
            package_name=package_name,
            module_id=module_id,
            config=config,
        ),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def generate_local_ai_sidecar(
    *,
    project_root: str | Path,
    mod_id: str,
    package_name: str,
    module: ProductionModule,
    policy: ScalePolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    module.validate(policy=policy)
    if module.kind != "integration":
        raise LocalAiSidecarGenerationError(
            "Local AI sidecar generation requires an integration module."
        )
    sidecar_policy = normalize_local_ai_sidecar_config(module.config)
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise LocalAiSidecarGenerationError(
            "Local AI sidecar target does not match fabric.mod.json."
        )

    source_path = local_ai_sidecar_source_path(package_name, module.module_id)
    manifest_path = local_ai_sidecar_manifest_path(module.module_id)
    source = render_local_ai_sidecar_source(
        package_name=package_name,
        module_id=module.module_id,
        config=sidecar_policy,
    )
    manifest = render_local_ai_sidecar_manifest(
        package_name=package_name,
        module_id=module.module_id,
        config=sidecar_policy,
    )
    patch = write_text_files(
        info,
        {
            source_path: source,
            manifest_path: manifest,
        },
        replace_existing=True,
    )
    return {
        "schema_version": "mmm/local-ai-sidecar-generation-v1",
        "status": "GENERATED",
        "module_id": module.module_id,
        "integration_type": INTEGRATION_TYPE,
        "endpoint": sidecar_policy.endpoint,
        "capabilities": list(sidecar_policy.capabilities),
        "files": [source_path, manifest_path],
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "manifest_sha256": sha256_bytes(manifest.encode("utf-8")),
        "policy_enforcement": "exact_source_and_manifest_reconstruction",
        "patch": patch,
    }


def _bounded_int(
    config: Mapping[str, Any],
    field: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = config.get(field, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise LocalAiSidecarGenerationError(
            f"{field} must be an integer from {minimum} through {maximum}."
        )
    return value
