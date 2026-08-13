from __future__ import annotations

import re
from pathlib import Path


DECODE = Path("minecraft_mod_ai/llama_decode_speed_contract.py")
ADAPTER_TEST = Path("tests/test_llama_cpp_adapter_request_contract.py")


def remove_decode_payload_wrapper() -> None:
    source = DECODE.read_text(encoding="utf-8")
    start = source.find("\ndef _install_host_validated_json_payload(")
    if start >= 0:
        end = source.find("\ndef ", start + 2)
        if end < 0:
            raise SystemExit("decode-speed payload wrapper has no following function anchor")
        source = source[:start] + source[end:]

    source = re.sub(
        r"(?m)^\s*_install_host_validated_json_payload\([^\n]*\)\s*\n",
        "",
        source,
    )
    if "_install_host_validated_json_payload" in source:
        raise SystemExit("decode-speed duplicate payload owner still remains")
    compile(source, str(DECODE), "exec")
    DECODE.write_text(source, encoding="utf-8")


def fix_error_body_privacy_test() -> None:
    source = ADAPTER_TEST.read_text(encoding="utf-8")
    old = '''                messages=({"role": "user", "content": "plan"},),\n                response_format="json",\n            )\n        )\n\n    message = str(caught.value)\n    assert "HTTP 400" in message\n    assert "unsupported request field" in message\n    assert "plan" not in message\n'''
    new = '''                messages=(\n                    {"role": "user", "content": "SECRET_PROMPT_SENTINEL"},\n                ),\n                response_format="json",\n            )\n        )\n\n    message = str(caught.value)\n    assert "HTTP 400" in message\n    assert "unsupported request field" in message\n    assert "SECRET_PROMPT_SENTINEL" not in message\n'''
    if new not in source:
        if old not in source:
            raise SystemExit("adapter privacy-test repair anchor not found")
        source = source.replace(old, new, 1)
    compile(source, str(ADAPTER_TEST), "exec")
    ADAPTER_TEST.write_text(source, encoding="utf-8")


def main() -> None:
    remove_decode_payload_wrapper()
    fix_error_body_privacy_test()


if __name__ == "__main__":
    main()
