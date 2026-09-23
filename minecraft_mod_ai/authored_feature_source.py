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
    r"@(?:net\.fabricmc\.api\.)?Environment\s*\(\s*(?:value\s*=\s*)?(?:(?:net\.fabricmc\.api\.)?EnvType\s*\.\s*)?(?:CLIENT|SERVER)\s*\)"
)


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
                if not body or re.fullmatch(r"(?:;|return;)+", body):
                    add(
                        "host:authored-empty",
                        "The authored initialize() has no executable behavior. Implement the approved unit.",
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
