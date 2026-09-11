from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "minecraft_mod_ai" / "data"


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _context_id(bundle: dict) -> str:
    raw = dict(bundle)
    raw.pop("context_id", None)
    return "sha256:" + sha256(_encode(raw).encode("utf-8")).hexdigest()


def main() -> None:
    catalog_path = DATA_DIR / "host_version_catalog.json"
    evidence_path = DATA_DIR / "official_version_evidence.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    old_auto = str(catalog.get("auto_context_id", ""))
    auto_index = 0
    by_minecraft: dict[str, str] = {}
    for index, bundle in enumerate(catalog["bundles"]):
        if bundle.get("context_id") == old_auto:
            auto_index = index
        context_id = _context_id(bundle)
        bundle["context_id"] = context_id
        minecraft = str(bundle["target"]["minecraft_version"])
        by_minecraft[minecraft] = context_id

    catalog["auto_context_id"] = catalog["bundles"][auto_index]["context_id"]
    for row in evidence.get("versions", []):
        minecraft = str(row.get("minecraft", ""))
        if minecraft in by_minecraft:
            row["context_id"] = by_minecraft[minecraft]

    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
