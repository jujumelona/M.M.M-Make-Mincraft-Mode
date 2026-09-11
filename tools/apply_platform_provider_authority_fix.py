from pathlib import Path

catalog = Path("minecraft_mod_ai/platform_catalog.py")
text = catalog.read_text(encoding="utf-8")
replacements = [
    (
        "    resolve: Callable[[str], TargetContract]\n",
        "    resolve: Callable[[str], TargetContract]\n    host_authoritative: bool = False\n",
    ),
    (
        "            resolve=provider.resolve,\n        )\n",
        "            resolve=provider.resolve,\n            host_authoritative=provider.host_authoritative,\n        )\n",
    ),
    (
        "        discover_versions=_fabric_versions,\n        resolve=_fabric_adapter,\n    )\n",
        "        discover_versions=_fabric_versions,\n        resolve=_fabric_adapter,\n        host_authoritative=True,\n    )\n",
    ),
]
for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"platform_catalog replacement expected once, found {count}: {old!r}")
    text = text.replace(old, new, 1)
catalog.write_text(text, encoding="utf-8")

pipeline = Path("minecraft_mod_ai/platform_selection_pipeline.py")
text = pipeline.read_text(encoding="utf-8")
old = '    if provider.provider_id == "host-coherent-version-catalog-v1":\n'
new = "    if provider.host_authoritative:\n"
count = text.count(old)
if count != 1:
    raise SystemExit(f"platform_selection replacement expected once, found {count}")
text = text.replace(old, new, 1)
pipeline.write_text(text, encoding="utf-8")
