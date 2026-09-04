from __future__ import annotations

import json

from minecraft_mod_ai.validation_diagnostic_contract import (
    diagnostic_errors,
    diagnostic_items,
    unwrap_diagnostic_receipt,
)


def _events(stderr: str) -> list[dict[str, object]]:
    prefix = "ROOT CAUSE TRACE: "
    return [
        json.loads(line[len(prefix) :])
        for line in stderr.splitlines()
        if line.startswith(prefix)
    ]


def test_structured_content_jdt_receipt_is_not_misclassified_unavailable(capsys) -> None:
    receipt = {
        "structured_content": {
            "status": "PASS",
            "diagnostics": {
                "file:///workspace/src/main/java/example/Ok.java": [],
            },
            "files_opened": 1,
        },
        "text": [],
        "parsed_text": None,
        "resources": [],
    }

    assert diagnostic_errors(receipt) == []

    events = _events(capsys.readouterr().err)
    layers = [event for event in events if event.get("event") == "diagnostic_receipt_layer"]
    assert layers
    assert layers[-1]["details"]["selected_key"] == "structured_content"
    classified = [event for event in events if event.get("event") == "diagnostic_receipt_classified"]
    assert classified
    assert classified[-1]["result"] == "PASS"
    assert classified[-1]["details"]["envelope_path"] == ["structured_content"]


def test_structured_content_preserves_real_java_error_instead_of_wrapper_error(capsys) -> None:
    uri = "file:///workspace/src/main/java/example/Broken.java"
    receipt = {
        "structured_content": {
            "status": "PASS",
            "diagnostics": {
                uri: [
                    {
                        "severity": 1,
                        "source": "jdtls",
                        "code": "compiler.err.expected",
                        "message": "Syntax error",
                    }
                ]
            },
        },
        "text": [],
        "parsed_text": None,
        "resources": [],
    }

    errors = diagnostic_errors(receipt)

    assert len(errors) == 1
    assert errors[0]["code"] == "compiler.err.expected"
    assert errors[0]["uri"] == uri
    assert all(error["code"] != "JDT_DIAGNOSTICS_UNAVAILABLE" for error in errors)

    events = _events(capsys.readouterr().err)
    classified = [event for event in events if event.get("event") == "diagnostic_receipt_classified"]
    assert classified[-1]["details"]["severity_1_count"] == 1


def test_structured_content_camel_case_is_supported(capsys) -> None:
    inner = {"status": "PASS", "diagnostics": {}}
    normalized, path = unwrap_diagnostic_receipt({"structuredContent": inner})

    assert normalized is inner
    assert path == ("structuredContent",)

    events = _events(capsys.readouterr().err)
    terminal = [event for event in events if event.get("event") == "diagnostic_receipt_unwrap"]
    assert terminal[-1]["result"] == "PASS"


def test_missing_structured_receipt_fails_closed_with_shape_evidence(capsys) -> None:
    receipt = {
        "structured_content": None,
        "text": ["not-json"],
        "parsed_text": None,
        "resources": [],
    }

    errors = diagnostic_errors(receipt)

    assert len(errors) == 1
    assert errors[0]["code"] == "JDT_DIAGNOSTICS_UNAVAILABLE"
    assert "missing or malformed" in errors[0]["message"]

    events = _events(capsys.readouterr().err)
    unwrap = [event for event in events if event.get("event") == "diagnostic_receipt_unwrap"]
    assert unwrap
    assert unwrap[-1]["result"] == "FAIL"
    assert "keys" in unwrap[-1]["details"]


def test_diagnostic_items_unwraps_transport_before_uri_normalization() -> None:
    uri = "file:///workspace/src/main/java/example/Ok.java"
    items = diagnostic_items(
        {
            "structured_content": {
                "status": "PASS",
                "diagnostics": {
                    uri: [
                        {
                            "severity": 2,
                            "message": "warning",
                        }
                    ]
                },
            }
        }
    )

    assert items == [{"severity": 2, "message": "warning", "uri": uri}]
