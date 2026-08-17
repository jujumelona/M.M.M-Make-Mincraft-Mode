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
INTEGRATION_TYPE = 'mmm_local_ai_sidecar'
TOKEN_SYSTEM_PROPERTY = 'mmm.sidecar.token'
TOKEN_ENVIRONMENT_VARIABLE = 'MMM_SIDECAR_TOKEN'
ALLOWED_CAPABILITIES = frozenset({'agent_tool_use', 'ai_inference', 'translation'})
_ALLOWED_CONFIG_FIELDS = frozenset({'integration_type', 'port', 'timeout_ms', 'max_request_bytes', 'max_response_bytes', 'max_in_flight', 'capabilities', 'authentication'})
_PACKAGE = re.compile('^[a-z_][a-z0-9_]*(?:\\.[a-z_][a-z0-9_]*)*$')
_MODULE_ID = re.compile('^[a-z][a-z0-9_\\-]{1,127}$')

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
        return f'http://127.0.0.1:{self.port}/v1/mmm/infer'

    def to_dict(self) -> dict[str, Any]:
        return {'integration_type': INTEGRATION_TYPE, 'port': self.port, 'timeout_ms': self.timeout_ms, 'max_request_bytes': self.max_request_bytes, 'max_response_bytes': self.max_response_bytes, 'max_in_flight': self.max_in_flight, 'capabilities': list(self.capabilities), 'authentication': self.authentication}

def normalize_local_ai_sidecar_config(config: Mapping[str, Any]) -> LocalAiSidecarPolicy:
    if not isinstance(config, Mapping):
        raise LocalAiSidecarGenerationError('Local AI sidecar config must be an object.')
    unknown = set(config) - _ALLOWED_CONFIG_FIELDS
    if unknown:
        raise LocalAiSidecarGenerationError('Unknown local AI sidecar config fields: ' + ', '.join(sorted(unknown)))
    if config.get('integration_type') != INTEGRATION_TYPE:
        raise LocalAiSidecarGenerationError(f'integration_type must equal {INTEGRATION_TYPE!r}.')
    port = _bounded_int(config, 'port', default=8765, minimum=1024, maximum=65535)
    timeout_ms = _bounded_int(config, 'timeout_ms', default=5000, minimum=100, maximum=30000)
    max_request_bytes = _bounded_int(config, 'max_request_bytes', default=262144, minimum=256, maximum=1048576)
    max_response_bytes = _bounded_int(config, 'max_response_bytes', default=262144, minimum=256, maximum=1048576)
    max_in_flight = _bounded_int(config, 'max_in_flight', default=4, minimum=1, maximum=32)
    raw_capabilities = config.get('capabilities')
    if type(raw_capabilities) is not list or not raw_capabilities:
        raise LocalAiSidecarGenerationError('capabilities must be a non-empty JSON array.')
    if any((not isinstance(value, str) for value in raw_capabilities)):
        raise LocalAiSidecarGenerationError('Every capability must be a string.')
    capabilities = tuple(sorted(set(raw_capabilities)))
    if len(capabilities) != len(raw_capabilities):
        raise LocalAiSidecarGenerationError('capabilities must not contain duplicates.')
    unsupported = set(capabilities) - ALLOWED_CAPABILITIES
    if unsupported:
        raise LocalAiSidecarGenerationError('Unsupported local AI sidecar capabilities: ' + ', '.join(sorted(unsupported)))
    authentication = config.get('authentication', 'none')
    if authentication not in {'none', 'external_token'}:
        raise LocalAiSidecarGenerationError("authentication must be 'none' or 'external_token'.")
    return LocalAiSidecarPolicy(port=port, timeout_ms=timeout_ms, max_request_bytes=max_request_bytes, max_response_bytes=max_response_bytes, max_in_flight=max_in_flight, capabilities=capabilities, authentication=str(authentication))

def local_ai_sidecar_source_path(package_name: str, module_id: str) -> str:
    class_name = local_ai_sidecar_class_name(module_id)
    return 'src/main/java/' + package_name.replace('.', '/') + f'/integration/{class_name}.java'

def local_ai_sidecar_manifest_path(module_id: str) -> str:
    return f'.minecraft_ai/integrations/{module_id}.local-ai-sidecar.json'

def local_ai_sidecar_class_name(module_id: str) -> str:
    if not _MODULE_ID.fullmatch(module_id):
        raise LocalAiSidecarGenerationError(f'Invalid sidecar module id: {module_id!r}')
    return ''.join((part.capitalize() for part in module_id.split('_'))) + 'LocalAiSidecar'

def render_local_ai_sidecar_source(*, package_name: str, module_id: str, config: Mapping[str, Any] | LocalAiSidecarPolicy) -> str:
    if not _PACKAGE.fullmatch(package_name):
        raise LocalAiSidecarGenerationError(f'Invalid Java package: {package_name!r}')
    policy = config if isinstance(config, LocalAiSidecarPolicy) else normalize_local_ai_sidecar_config(config)
    class_name = local_ai_sidecar_class_name(module_id)
    capabilities = ',\n            '.join((f'Capability.{value.upper()}' for value in policy.capabilities))
    enum_values = ',\n        '.join((f'{value.upper()}("{value}")' for value in sorted(ALLOWED_CAPABILITIES)))
    auth_required = 'true' if policy.authentication == 'external_token' else 'false'
    return f'''package {package_name}.integration;\n\nimport com.google.gson.JsonElement;\nimport com.google.gson.JsonObject;\nimport com.google.gson.JsonParser;\nimport java.io.ByteArrayOutputStream;\nimport java.net.URI;\nimport java.net.http.HttpClient;\nimport java.net.http.HttpRequest;\nimport java.net.http.HttpResponse;\nimport java.nio.ByteBuffer;\nimport java.nio.charset.StandardCharsets;\nimport java.time.Duration;\nimport java.util.List;\nimport java.util.Set;\nimport java.util.concurrent.CompletableFuture;\nimport java.util.concurrent.CompletionStage;\nimport java.util.concurrent.Flow;\nimport java.util.concurrent.Semaphore;\n\n/**\n * Generated, fail-closed boundary for one reviewed localhost AI sidecar.\n * This utility returns typed data only and never mutates Minecraft world state.\n */\npublic final class {class_name} {{\n    private static final URI ENDPOINT = URI.create("{policy.endpoint}");\n    private static final int TIMEOUT_MS = {policy.timeout_ms};\n    private static final int MAX_REQUEST_BYTES = {policy.max_request_bytes};\n    private static final int MAX_RESPONSE_BYTES = {policy.max_response_bytes};\n    private static final boolean AUTHENTICATION_REQUIRED = {auth_required};\n    private static final Set<Capability> ENABLED_CAPABILITIES = Set.of(\n            {capabilities}\n    );\n    private static final Semaphore IN_FLIGHT = new Semaphore({policy.max_in_flight});\n    private static final HttpClient CLIENT = HttpClient.newBuilder()\n            .connectTimeout(Duration.ofMillis(TIMEOUT_MS))\n            .followRedirects(HttpClient.Redirect.NEVER)\n            .version(HttpClient.Version.HTTP_1_1)\n            .build();\n\n    private {class_name}() {{\n    }}\n\n    public static CompletableFuture<InferenceResponse> infer(\n            String requestId,\n            Capability capability,\n            JsonObject input\n    ) {{\n        if (!isValidRequestId(requestId)) {{\n            return failed("requestId must use 1-128 safe identifier characters");\n        }}\n        if (capability == null || !ENABLED_CAPABILITIES.contains(capability)) {{\n            return failed("capability is not enabled by the approved policy");\n        }}\n\n        JsonObject request = new JsonObject();\n        request.addProperty("schema_version", "mmm/sidecar-inference-request-v1");\n        request.addProperty("request_id", requestId);\n        request.addProperty("capability", capability.wireName());\n        request.add("input", input == null ? new JsonObject() : input.deepCopy());\n        byte[] requestBytes = request.toString().getBytes(StandardCharsets.UTF_8);\n        if (requestBytes.length > MAX_REQUEST_BYTES) {{\n            return failed("request exceeds the approved byte limit");\n        }}\n\n        String token;\n        try {{\n            token = externalToken();\n        }} catch (SidecarException error) {{\n            return CompletableFuture.failedFuture(error);\n        }}\n        if (!IN_FLIGHT.tryAcquire()) {{\n            return failed("sidecar concurrency limit reached");\n        }}\n\n        HttpRequest.Builder builder = HttpRequest.newBuilder(ENDPOINT)\n                .timeout(Duration.ofMillis(TIMEOUT_MS))\n                .header("Content-Type", "application/json")\n                .header("Accept", "application/json");\n        if (token != null) {{\n            builder.header("Authorization", "Bearer " + token);\n        }}\n        HttpRequest httpRequest;\n        try {{\n            httpRequest = builder.POST(HttpRequest.BodyPublishers.ofByteArray(requestBytes)).build();\n        }} catch (RuntimeException error) {{\n            IN_FLIGHT.release();\n            return CompletableFuture.failedFuture(error);\n        }}\n\n        try {{\n            return CLIENT.sendAsync(httpRequest, {class_name}::boundedBody)\n                    .thenApplyAsync(response -> parseResponse(requestId, response))\n                    .whenComplete((unused, error) -> IN_FLIGHT.release());\n        }} catch (RuntimeException error) {{\n            IN_FLIGHT.release();\n            return CompletableFuture.failedFuture(error);\n        }}\n    }}\n\n    private static InferenceResponse parseResponse(\n            String expectedRequestId,\n            HttpResponse<byte[]> response\n    ) {{\n        try {{\n            if (response.statusCode() != 200) {{\n                throw new SidecarException("sidecar returned HTTP " + response.statusCode());\n            }}\n            byte[] bytes = response.body();\n            JsonElement parsed = JsonParser.parseString(new String(bytes, StandardCharsets.UTF_8));\n            if (!parsed.isJsonObject()) {{\n                throw new SidecarException("sidecar response must be a JSON object");\n            }}\n            return typedResponse(expectedRequestId, parsed.getAsJsonObject());\n        }} catch (RuntimeException error) {{\n            if (error instanceof SidecarException sidecarError) {{\n                throw sidecarError;\n            }}\n            throw new SidecarException("sidecar response could not be decoded", error);\n        }}\n    }}\n\n    private static HttpResponse.BodySubscriber<byte[]> boundedBody(\n            HttpResponse.ResponseInfo responseInfo\n    ) {{\n        long declaredLength = responseInfo.headers()\n                .firstValueAsLong("Content-Length")\n                .orElse(-1L);\n        return new BoundedBodySubscriber(MAX_RESPONSE_BYTES, declaredLength);\n    }}\n\n    private static InferenceResponse typedResponse(\n            String expectedRequestId,\n            JsonObject response\n    ) {{\n        Set<String> allowedFields = Set.of(\n                "schema_version", "request_id", "status", "output", "error"\n        );\n        if (!allowedFields.containsAll(response.keySet())) {{\n            throw new SidecarException("sidecar response contains unknown fields");\n        }}\n        String schemaVersion = requiredString(response, "schema_version", 64);\n        String requestId = requiredString(response, "request_id", 128);\n        String status = requiredString(response, "status", 16);\n        if (!"mmm/sidecar-inference-response-v1".equals(schemaVersion)) {{\n            throw new SidecarException("sidecar response schema is unsupported");\n        }}\n        if (!expectedRequestId.equals(requestId)) {{\n            throw new SidecarException("sidecar response request_id mismatch");\n        }}\n        if (!status.equals("ok") && !status.equals("error")) {{\n            throw new SidecarException("sidecar response status is invalid");\n        }}\n        JsonObject output = optionalObject(response, "output");\n        JsonObject error = optionalObject(response, "error");\n        String errorCode = optionalString(error, "code", 128);\n        String errorMessage = optionalString(error, "message", 1024);\n        if (status.equals("ok") && error != null) {{\n            throw new SidecarException("successful sidecar response may not contain error");\n        }}\n        if (status.equals("error") && (errorCode == null || errorMessage == null)) {{\n            throw new SidecarException("error response requires typed code and message");\n        }}\n        return new InferenceResponse(\n                requestId,\n                status,\n                output == null ? new JsonObject() : output,\n                errorCode,\n                errorMessage\n        );\n    }}\n\n    private static String requiredString(JsonObject object, String key, int maxLength) {{\n        String value = optionalString(object, key, maxLength);\n        if (value == null || value.isBlank()) {{\n            throw new SidecarException("sidecar response is missing " + key);\n        }}\n        return value;\n    }}\n\n    private static String optionalString(JsonObject object, String key, int maxLength) {{\n        if (object == null || !object.has(key) || object.get(key).isJsonNull()) {{\n            return null;\n        }}\n        JsonElement value = object.get(key);\n        if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {{\n            throw new SidecarException(key + " must be a string");\n        }}\n        String text = value.getAsString();\n        if (text.length() > maxLength) {{\n            throw new SidecarException(key + " exceeds its character limit");\n        }}\n        return text;\n    }}\n\n    private static JsonObject optionalObject(JsonObject object, String key) {{\n        if (!object.has(key) || object.get(key).isJsonNull()) {{\n            return null;\n        }}\n        if (!object.get(key).isJsonObject()) {{\n            throw new SidecarException(key + " must be an object");\n        }}\n        return object.getAsJsonObject(key).deepCopy();\n    }}\n\n    private static boolean isValidRequestId(String value) {{\n        if (value == null || value.isEmpty() || value.length() > 128) {{\n            return false;\n        }}\n        for (int index = 0; index < value.length(); index++) {{\n            char character = value.charAt(index);\n            boolean safe = Character.isLetterOrDigit(character)\n                    || character == '.' || character == '_' || character == ':' || character == '-';\n            if (!safe) {{\n                return false;\n            }}\n        }}\n        return true;\n    }}\n\n    private static String externalToken() {{\n        if (!AUTHENTICATION_REQUIRED) {{\n            return null;\n        }}\n        String token = System.getProperty("{TOKEN_SYSTEM_PROPERTY}");\n        if (token == null || token.isBlank()) {{\n            token = System.getenv("{TOKEN_ENVIRONMENT_VARIABLE}");\n        }}\n        if (token == null || token.isBlank()) {{\n            throw new SidecarException("external sidecar token is required");\n        }}\n        if (token.length() > 512 || token.indexOf('\\r') >= 0 || token.indexOf('\\n') >= 0) {{\n            throw new SidecarException("external sidecar token is invalid");\n        }}\n        return token;\n    }}\n\n    private static <T> CompletableFuture<T> failed(String message) {{\n        return CompletableFuture.failedFuture(new SidecarException(message));\n    }}\n\n    private static final class BoundedBodySubscriber\n            implements HttpResponse.BodySubscriber<byte[]> {{\n        private final int limit;\n        private final ByteArrayOutputStream buffer;\n        private final CompletableFuture<byte[]> body = new CompletableFuture<>();\n        private final boolean declaredTooLarge;\n        private Flow.Subscription subscription;\n\n        private BoundedBodySubscriber(int limit, long declaredLength) {{\n            this.limit = limit;\n            this.buffer = new ByteArrayOutputStream(Math.min(limit, 8192));\n            this.declaredTooLarge = declaredLength > limit;\n            if (declaredTooLarge) {{\n                body.completeExceptionally(\n                        new SidecarException("response exceeds the approved byte limit")\n                );\n            }}\n        }}\n\n        @Override\n        public CompletionStage<byte[]> getBody() {{\n            return body;\n        }}\n\n        @Override\n        public void onSubscribe(Flow.Subscription value) {{\n            if (subscription != null) {{\n                value.cancel();\n                return;\n            }}\n            subscription = value;\n            if (declaredTooLarge) {{\n                value.cancel();\n            }} else {{\n                value.request(1);\n            }}\n        }}\n\n        @Override\n        public void onNext(List<ByteBuffer> chunks) {{\n            if (body.isDone()) {{\n                subscription.cancel();\n                return;\n            }}\n            for (ByteBuffer chunk : chunks) {{\n                int length = chunk.remaining();\n                if (length > limit - buffer.size()) {{\n                    subscription.cancel();\n                    body.completeExceptionally(\n                            new SidecarException("response exceeds the approved byte limit")\n                    );\n                    return;\n                }}\n                byte[] bytes = new byte[length];\n                chunk.get(bytes);\n                buffer.write(bytes, 0, bytes.length);\n            }}\n            subscription.request(1);\n        }}\n\n        @Override\n        public void onError(Throwable error) {{\n            body.completeExceptionally(error);\n        }}\n\n        @Override\n        public void onComplete() {{\n            body.complete(buffer.toByteArray());\n        }}\n    }}\n\n    public enum Capability {{\n        {enum_values};\n\n        private final String wireName;\n\n        Capability(String wireName) {{\n            this.wireName = wireName;\n        }}\n\n        public String wireName() {{\n            return wireName;\n        }}\n    }}\n\n    public record InferenceResponse(\n            String requestId,\n            String status,\n            JsonObject output,\n            String errorCode,\n            String errorMessage\n    ) {{\n        public InferenceResponse {{\n            output = output.deepCopy();\n        }}\n\n        @Override\n        public JsonObject output() {{\n            return output.deepCopy();\n        }}\n    }}\n\n    public static final class SidecarException extends RuntimeException {{\n        public SidecarException(String message) {{\n            super(message);\n        }}\n\n        public SidecarException(String message, Throwable cause) {{\n            super(message, cause);\n        }}\n    }}\n}}\n'''

def build_local_ai_sidecar_manifest(*, package_name: str, module_id: str, config: Mapping[str, Any] | LocalAiSidecarPolicy) -> dict[str, Any]:
    policy = config if isinstance(config, LocalAiSidecarPolicy) else normalize_local_ai_sidecar_config(config)
    source_path = local_ai_sidecar_source_path(package_name, module_id)
    source = render_local_ai_sidecar_source(package_name=package_name, module_id=module_id, config=policy)
    return {'schema_version': 'mmm/local-ai-sidecar-policy-v1', 'module_id': module_id, 'source': {'path': source_path, 'sha256': sha256_bytes(source.encode('utf-8')), 'validation': 'exact_reconstruction_required'}, 'network': {'endpoint': policy.endpoint, 'method': 'POST', 'redirects': 'disabled', 'transport': 'java_17_httpclient_send_async', 'timeout_ms': policy.timeout_ms, 'max_request_bytes': policy.max_request_bytes, 'max_response_bytes': policy.max_response_bytes, 'max_in_flight': policy.max_in_flight}, 'capabilities': list(policy.capabilities), 'authority': {'returns_typed_json_only': True, 'minecraft_world_mutation': 'none', 'synchronous_wait': 'forbidden'}, 'secrets': {'embedded': False, 'authentication': policy.authentication, 'system_property': TOKEN_SYSTEM_PROPERTY if policy.authentication == 'external_token' else None, 'environment_variable': TOKEN_ENVIRONMENT_VARIABLE if policy.authentication == 'external_token' else None}, 'approved_config': policy.to_dict()}

def render_local_ai_sidecar_manifest(*, package_name: str, module_id: str, config: Mapping[str, Any] | LocalAiSidecarPolicy) -> str:
    return json.dumps(build_local_ai_sidecar_manifest(package_name=package_name, module_id=module_id, config=config), ensure_ascii=False, indent=2, sort_keys=True) + '\n'

def generate_local_ai_sidecar(*, project_root: str | Path, mod_id: str, package_name: str, module: ProductionModule, policy: ScalePolicy | None=None) -> dict[str, Any]:
    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    module.validate(policy=policy)
    if module.kind != 'integration':
        raise LocalAiSidecarGenerationError('Local AI sidecar generation requires an integration module.')
    sidecar_policy = normalize_local_ai_sidecar_config(module.config)
    info = inspect_fabric_project(project_root)
    if info.mod_id != mod_id or info.package_name != package_name:
        raise LocalAiSidecarGenerationError('Local AI sidecar target does not match fabric.mod.json.')
    source_path = local_ai_sidecar_source_path(package_name, module.module_id)
    manifest_path = local_ai_sidecar_manifest_path(module.module_id)
    source = render_local_ai_sidecar_source(package_name=package_name, module_id=module.module_id, config=sidecar_policy)
    manifest = render_local_ai_sidecar_manifest(package_name=package_name, module_id=module.module_id, config=sidecar_policy)
    patch = write_text_files(info, {source_path: source, manifest_path: manifest}, replace_existing=True)
    return {'schema_version': 'mmm/local-ai-sidecar-generation-v1', 'status': 'GENERATED', 'module_id': module.module_id, 'integration_type': INTEGRATION_TYPE, 'endpoint': sidecar_policy.endpoint, 'capabilities': list(sidecar_policy.capabilities), 'files': [source_path, manifest_path], 'source_sha256': sha256_bytes(source.encode('utf-8')), 'manifest_sha256': sha256_bytes(manifest.encode('utf-8')), 'policy_enforcement': 'exact_source_and_manifest_reconstruction', 'patch': patch}

def _bounded_int(config: Mapping[str, Any], field: str, *, default: int, minimum: int, maximum: int) -> int:
    value = config.get(field, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise LocalAiSidecarGenerationError(f'{field} must be an integer from {minimum} through {maximum}.')
    return value
