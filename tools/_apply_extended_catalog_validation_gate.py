from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding='utf-8')
    if new in text:
        return
    if old not in text:
        raise SystemExit(f'{label}: patch anchor missing')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')


def main() -> None:
    validator = Path('minecraft_mod_ai/validator.py')
    replace_once(
        validator,
        'from .complete_spec import CompleteProposal\n',
        'from .complete_spec import CompleteProposal\nfrom .extended_content_generator import ExtendedContentError, iter_extended_module_records\n',
        'validator import',
    )
    replace_once(
        validator,
        '''        reviewed_network_sources = _reviewed_local_ai_sidecar_network_sources(\n            root,\n            spec,\n            complete,\n        )\n\n        for path in sorted(root.rglob("*")):\n''',
        '''        reviewed_network_sources = _reviewed_local_ai_sidecar_network_sources(\n            root,\n            spec,\n            complete,\n        )\n\n        extended_catalog = root / ".minecraft_ai/extended-modules.json"\n        if extended_catalog.is_file() and not extended_catalog.is_symlink():\n            checks += 1\n            try:\n                tuple(iter_extended_module_records(root))\n            except (ExtendedContentError, OSError, UnicodeError, json.JSONDecodeError) as exc:\n                findings.append(\n                    Finding(\n                        "EXTENDED_CATALOG_INVALID",\n                        "error",\n                        self._rel(root, extended_catalog),\n                        f"Extended content catalog failed structural validation: {type(exc).__name__}: {exc}",\n                    )\n                )\n\n        for path in sorted(root.rglob("*")):\n''',
        'validation gate',
    )

    test = Path('tests/test_extended_catalog_validation_gate.py')
    test.write_text('''from __future__ import annotations\n\nimport json\nfrom pathlib import Path\n\nfrom minecraft_mod_ai import validator as validator_module\nfrom minecraft_mod_ai.extended_content_generator import ExtendedContentError\nfrom minecraft_mod_ai.generator import FabricProjectGenerator\nfrom minecraft_mod_ai.spec import ContentKind, ContentSpec, ModSpec\nfrom minecraft_mod_ai.validator import ProjectValidator\n\n\ndef _spec() -> ModSpec:\n    return ModSpec(\n        mod_id="catalog_gate",\n        mod_name="Catalog Gate",\n        package_name="ai.minecraft.catalog_gate",\n        version="1.0.0",\n        summary="catalog validation gate",\n        contents=(\n            ContentSpec(\n                content_id="core_item",\n                kind=ContentKind.ITEM,\n                display_name_en="Core Item",\n                display_name_ko="핵심 아이템",\n            ),\n        ),\n    )\n\n\ndef _project(tmp_path: Path) -> tuple[Path, ModSpec]:\n    spec = _spec()\n    root = tmp_path / "project"\n    FabricProjectGenerator().generate(spec, root)\n    metadata = root / ".minecraft_ai"\n    metadata.mkdir(parents=True, exist_ok=True)\n    (metadata / "extended-modules.json").write_text(\n        json.dumps(\n            {\n                "schema_version": "mmm/extended-module-directory-v1",\n                "module_count": 0,\n                "record_directory": ".minecraft_ai/extended-module-records",\n            }\n        ),\n        encoding="utf-8",\n    )\n    return root, spec\n\n\ndef test_extended_catalog_is_structurally_checked_once_per_validation(tmp_path: Path, monkeypatch) -> None:\n    root, spec = _project(tmp_path)\n    calls: list[Path] = []\n\n    def fake_iter(project_root):\n        calls.append(Path(project_root))\n        return iter(())\n\n    monkeypatch.setattr(validator_module, "iter_extended_module_records", fake_iter)\n    ProjectValidator().validate(root, spec)\n    assert calls == [root.resolve()]\n\n\ndef test_extended_catalog_structural_failure_is_fail_closed(tmp_path: Path, monkeypatch) -> None:\n    root, spec = _project(tmp_path)\n\n    def broken_iter(project_root):\n        raise ExtendedContentError("module_count mismatch")\n\n    monkeypatch.setattr(validator_module, "iter_extended_module_records", broken_iter)\n    report = ProjectValidator().validate(root, spec)\n    findings = [finding for finding in report.findings if finding.code == "EXTENDED_CATALOG_INVALID"]\n    assert len(findings) == 1\n    assert findings[0].path == ".minecraft_ai/extended-modules.json"\n    assert "module_count mismatch" in findings[0].message\n    assert report.status == "FAIL"\n''', encoding='utf-8')


if __name__ == '__main__':
    main()
