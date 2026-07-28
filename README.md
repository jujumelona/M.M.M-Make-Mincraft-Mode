# M.M.M Make Mincraft Mode

**English** | [한국어](README_KO.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

Create Minecraft Java Fabric 1.20.1 mods from a plain-language request, then
download the built and verified JAR and release ZIP.

> New mods do not require a ZIP upload. Upload a ZIP only when modifying an
> existing mod.

## Quick start

1. Click **Open In Colab** above.
2. In Colab, select `Runtime → Run all`.
3. Enter the mod you want in the screen opened by the last cell.
4. Click `1. Create plan`, then paste the displayed approval hash into the field below it.
5. Click `2. Approve and run`.
6. Download the **release ZIP** at the bottom when the build finishes.

Example request:

```text
Create an ice-magic boss, a battle arena map, a 3D model,
crystal items, and crystal blocks.
```

### Modify an existing mod

1. Change `PATCH_EXISTING = False` to `True` in Colab.
2. Upload one source/release ZIP that you have permission to modify.
3. Follow the same steps as for a new mod.

## What it can create

- Items, blocks, recipes, loot tables, tags, and Korean/English names
- Bosses, boss bars, spawn eggs, dedicated items, and loot
- Command-installed battle arena maps and datapacks
- Entity textures, Blockbench `.bbmodel`, and `.obj/.mtl` files
- Fabric source projects, verified JARs, and release ZIPs

The fixed environment is Minecraft `1.20.1`, Java `17`, and Fabric.

## Output

The release ZIP has this structure:

```text
art_sources/   3D models and textures
binaries/      verified installable JAR
docs/          installation and administrator guides
evidence/      build and validation results
packs/         arena datapack
source/        generated Fabric source ZIP
world/         map design JSON and preview
```

If the build or validation fails, no installable JAR is placed in `binaries/`.

## Run locally

Python 3.10 or later and Java 17 are required.

```powershell
python -m pip install -e ".[ui]"
python -m minecraft_mod_ai.cli ui
```

Tests:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Project structure

```text
M.M.M_Make_Mincraft_Mode_Colab.ipynb  Colab notebook
minecraft_mod_ai/                     mod generation and build program
tests/                                automated tests
tools/                                Colab and build tools
```

## License

This project uses the [MIT License](LICENSE). Commercial use, modification, and
redistribution are allowed. Distributions must include the copyright notice and
license text. The software is provided without warranty; see `LICENSE` for the
complete terms.
