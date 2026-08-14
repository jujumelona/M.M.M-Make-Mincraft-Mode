from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: annotate_pytest_failures.py <junit.xml>")
    path = Path(sys.argv[1])
    if not path.is_file():
        print("::error title=pytest::JUnit XML missing", flush=True)
        return 1
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"::error title=pytest::Invalid JUnit XML: {_escape(str(exc))}", flush=True)
        return 1

    count = 0
    for case in root.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        node = failure if failure is not None else error
        if node is None:
            continue
        count += 1
        classname = str(case.attrib.get("classname", "pytest"))
        name = str(case.attrib.get("name", "unknown"))
        message = str(node.attrib.get("message", "")) or (node.text or "test failed")
        message = " ".join(message.split())[:1600]
        title = _escape(f"{classname}::{name}")
        print(f"::error title={title}::{_escape(message)}", flush=True)
    if not count:
        print("::error title=pytest::Test process failed without JUnit failure nodes", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
