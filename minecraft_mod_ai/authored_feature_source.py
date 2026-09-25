from __future__ import annotations

"""Source obligations of host-wired authored feature entry points.

These checks establish integration, not arbitrary gameplay correctness. Client-only
helpers remain allowed, but the class and initialize() called by the common host
entry point must survive both Fabric environments.
"""

import re

_NON_CODE = re.compile(
    r'""".*?"""|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*.*?\*/', re.DOTALL
)
_INITIALIZE = re.compile(r"\bpublic\s+static\s+void\s+initialize\s*\(\s*\)\s*\{")
_SIDE_ONLY = re.compile(
    r"@(?:net\.fabricmc\.api\.)?Environment\s*\(\s*(?:value\s*=\s*)?(?:(?:net\.fabricmc\.api\.)?EnvType\s*\.\s*)?(?P<side>CLIENT|SERVER)\s*\)"
)
_STATIC_VOID_METHOD = re.compile(
    r"\b(?:public\s+|protected\s+|private\s+)?static\s+void\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\([^{};]*\)\s*\{"
)


def _only_private_initialization_guard(code: str, body_start: int, body_end: int) -> bool:
    """Recognize inert latches without guessing whether arbitrary Java is gameplay.

    Accept only private nonvolatile boolean fields and private Object monitors
    used exclusively by this initializer. Any other operation or external use
    leaves judgment to the compiler and subsequent behavior gates.
    """
    outside = code[:body_start] + code[body_end:]
    guard_field = re.compile(
        r"(?<=[;{}])\s*private\s+static\s+boolean\s+([A-Za-z_$][\w$]*)\s*(?:=\s*(?:true|false))?\s*;"
    )
    lock_field = re.compile(
        r"(?<=[;{}])\s*private\s+static\s+final\s+(?:java\.lang\.)?Object\s+([A-Za-z_$][\w$]*)"
        r"\s*=\s*new\s+(?:java\.lang\.)?Object\s*\(\s*\)\s*;"
    )
    guards = [match.group(1) for match in guard_field.finditer(outside)]
    locks = [match.group(1) for match in lock_field.finditer(outside)]
    remaining = lock_field.sub("", guard_field.sub("", outside))
    guards = [name for name in guards if not re.search(rf"\b{re.escape(name)}\b", remaining)]
    locks = [name for name in locks if not re.search(rf"\b{re.escape(name)}\b", remaining)]
    if not guards:
        return False
    body = code[body_start:body_end]
    names = "(?:" + "|".join(re.escape(name) for name in guards) + ")"
    body = re.sub(rf"\bif\s*\(\s*!?\s*{names}\s*\)", "", body)
    body = re.sub(rf"\b{names}\s*=\s*(?:true|false)\s*;", "", body)
    for name in locks:
        body = re.sub(rf"\bsynchronized\s*\(\s*{re.escape(name)}\s*\)", "", body)
    body = re.sub(r"\breturn\s*;|\belse\b", "", body)
    return not re.sub(r"[\s{};]", "", body)


def _matching_brace(code: str, open_brace: int) -> int | None:
    depth = 0
    for index in range(open_brace, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _side_only_static_void_methods(
    code: str, *, class_body_start: int, class_body_end: int
) -> dict[str, tuple[str, int]]:
    """Return side-stripped static void helpers declared directly on the feature class."""

    methods: dict[str, tuple[str, int]] = {}
    depth = 1
    index = class_body_start
    while index < class_body_end:
        match = _STATIC_VOID_METHOD.match(code, index) if depth == 1 else None
        if match is not None:
            prefix_start = (
                max(code.rfind(delimiter, class_body_start, match.start()) for delimiter in (";", "{", "}"))
                + 1
            )
            annotation = _SIDE_ONLY.search(code[prefix_start:match.start()])
            if annotation is not None:
                methods[match.group("name")] = (
                    annotation.group("side"),
                    prefix_start + annotation.start(),
                )
            body_end = _matching_brace(code, match.end() - 1)
            if body_end is not None:
                index = body_end + 1
                continue
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
        index += 1
    return methods


def authored_feature_source_diagnostics(
    source: str, *, path: str, symbol: str
) -> list[dict]:
    # Mask literals/comments in one lexical pass, retaining offsets and line numbers.
    code = _NON_CODE.sub(lambda match: re.sub(r"[^\r\n]", " ", match.group()), source)
    findings: list[dict] = []

    def add(code_id: str, message: str, offset: int = 0) -> None:
        findings.append(
            {
                "path": path,
                "line": source.count("\n", 0, offset) + 1,
                "severity": 1,
                "source": "host-authored-contract",
                "code": code_id,
                "message": message,
            }
        )

    declaration = re.search(
        rf"\bpublic\s+final\s+class\s+{re.escape(symbol)}\b[^{{;]*\{{", code
    )
    if declaration is None:
        add(
            "host:authored-surface",
            f"Host integration requires public final class {symbol}.",
        )
        return findings
    start = declaration.end()
    depth = 1
    method = None
    for index in range(start, len(code)):
        if depth == 1 and (match := _INITIALIZE.match(code, index)):
            method = match
            break
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                break
    if method is None:
        add(
            "host:authored-surface",
            "Host integration requires public static void initialize() on the feature class.",
            declaration.start(),
        )
    for member in (declaration, method):
        if member is None:
            continue
        prefix_start = (
            max(
                code.rfind(delimiter, 0, member.start())
                for delimiter in (";", "{", "}")
            )
            + 1
        )
        side_annotation = _SIDE_ONLY.search(code[prefix_start : member.start()])
        if side_annotation:
            add(
                "host:authored-side-only",
                "Host integration calls initialize() on both client and server. Keep its declaring class and method available on both sides; isolate side-specific behavior in helpers instead of annotating the common surface.",
                prefix_start + side_annotation.start(),
            )
    if method is not None:
        class_end = _matching_brace(code, declaration.end() - 1)
        initialize_end = _matching_brace(code, method.end() - 1)
        if class_end is not None and initialize_end is not None:
            side_helpers = _side_only_static_void_methods(
                code,
                class_body_start=declaration.end(),
                class_body_end=class_end,
            )
            initialize_body = code[method.end():initialize_end]
            for helper_name, (side, _annotation_offset) in side_helpers.items():
                if helper_name == "initialize":
                    continue
                call = re.search(
                    rf"(?<![\w$])(?:{re.escape(symbol)}\s*\.\s*)?"
                    rf"{re.escape(helper_name)}\s*\(",
                    initialize_body,
                )
                if call is not None:
                    add(
                        "host:authored-side-call",
                        (
                            f"Host common initialize() directly calls @Environment({side}) "
                            f"helper {helper_name}(). Fabric strips that helper on the opposite "
                            "runtime side, which can produce NoSuchMethodError. Keep common "
                            "initialize() side-neutral and wire side-only helpers from a matching "
                            "side-specific entry point."
                        ),
                        method.end() + call.start(),
                    )

    for match in _NON_CODE.finditer(source):
        if (
            match.group().startswith(("//", "/*"))
            and "MMM_AUTHORED_FEATURE_BODY_" in match.group()
        ):
            add(
                "host:authored-placeholder",
                "The authored feature still contains the host's unimplemented body marker. Implement the approved unit and replace that marker; initialization guards alone do not implement the unit.",
                match.start(),
            )
    if method is not None:
        depth = 1
        for index in range(method.end(), len(code)):
            depth += (code[index] == "{") - (code[index] == "}")
            if depth == 0:
                body = re.sub(r"\s+", "", code[method.end() : index])
                if (not body or re.fullmatch(r"(?:;|return;)+", body)
                        or _only_private_initialization_guard(code, method.end(), index)):
                    add(
                        "host:authored-empty",
                        "The authored initialize() has no executable behavior beyond an empty body or private initialization guard. Implement the approved unit; removing the marker or setting an initialization flag alone is insufficient.",
                        method.start(),
                    )
                break
    return findings


def current_authored_feature_diagnostics(root, target: str) -> list[dict]:
    from .small_model_task_capsule_contract import _CURRENT_CAPSULE

    capsule = _CURRENT_CAPSULE.get()
    if (
        capsule is None
        or capsule.primary_path != target
        or re.fullmatch(r"authored_feature_\d+", capsule.task_id) is None
    ):
        return []
    candidate = root / target
    try:
        candidate.resolve().relative_to(root)
        if candidate.is_symlink():
            return []  # Mutation authority owns path admission.
        source = candidate.read_text(encoding="utf-8")
    except (OSError, ValueError, UnicodeError):
        return []  # The compiler retains authority over missing/unreadable source.
    return authored_feature_source_diagnostics(
        source, path=target, symbol=capsule.primary_symbol
    )
