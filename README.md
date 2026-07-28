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
3. Tell the AI what kind of mod you want.
4. Read its plain-language plan and continue chatting to change anything.
5. Click `이대로 만들기` (**Build this plan**), or type `go ahead`, when the plan is correct.
6. Download the **release ZIP** when the build finishes.

Anything you did not request is not added automatically.
Small mods receive a concise production plan. Large projects include the core
gameplay loop, world/level design, systems/content, production milestones, and
release criteria. The AI asks before deciding an unclear project scale or first
playable scope.

Example request:

```text
Create two maple-themed items, three blocks, and a 41x41 arena.
Do not add a boss.
```

### Modify an existing mod

1. Change `PATCH_EXISTING = False` to `True` in Colab.
2. Upload one source/release ZIP that you have permission to modify.
3. Follow the same steps as for a new mod.

## The notebook is only a launcher

The repository package contains the mod planner, generator, validator, and
builder. The provided notebook only installs the current package from GitHub and
opens its interface; it is not a separate or frozen copy of the engine.

You can therefore use the package in any new Google Colab notebook:

```python
%pip install -q --upgrade "mmm-make-mincraft-mode[ui] @ git+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git@main"

from minecraft_mod_ai.webui import launch

launch(output_root="/content/mmm-output", share=True)
```

Running the installation cell again obtains the current `main` version from
GitHub.

## Python API

`ModAISession` provides a small stateful API for planning, revising, and
building without the notebook UI:

```python
from minecraft_mod_ai import ModAISession, supported_minecraft_versions

print(supported_minecraft_versions())  # ('1.20.1',)

session = ModAISession(
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
)

first = session.plan("Create one frost item.")
print(first.message)

revised = session.revise("Also add one frost block.")
print(revised.message)

if revised.ready_to_build:  # revised.buildable is an equivalent alias
    result = session.build(source_only=False)
    print(result.release_zip)
```

- `plan(request)` starts a new draft and returns a conversational reply.
- `revise(change)` updates the current draft and returns the revised reply.
- `reply.message`, `reply.questions`, and `reply.ready_to_build` describe what
  is ready or still needs clarification.
- `build(source_only=False)` performs the build and returns the release result.
  Use `source_only=True` when you only need the generated project.

### Optional OpenAI-compatible API

The built-in planner is the default. To use an external
OpenAI-compatible HTTPS chat-completions endpoint in Colab, add a private Colab
Secret named `MMM_API_KEY` and allow the notebook to access it. Never paste the
key into a notebook cell, prompt, repository file, or shared link.

```python
from google.colab import userdata

from minecraft_mod_ai import ModAISession

api_key = userdata.get("MMM_API_KEY")
if not api_key:
    raise RuntimeError("Add the MMM_API_KEY Colab Secret and allow notebook access.")

session = ModAISession.with_openai_compatible_api(
    base_url="https://your-provider.example/v1",
    model="your-model-name",
    api_key=api_key,
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
)
```

In the provided launcher notebook, select `api`, enter the HTTPS base URL and
model name, and keep the key only in the `MMM_API_KEY` Secret.

## Current build output

- Items, blocks, recipes, loot tables, tags, and Korean/English names
- Boss entities, spawn eggs, and loot only when explicitly requested
- Explicitly requested command-installed arena maps and datapacks
- Entity textures, Blockbench `.bbmodel`, and `.obj/.mtl` files
- Fabric source projects, verified JARs, and release ZIPs

## Supported Minecraft target

The currently supported target is exactly:

- Minecraft Java Edition `1.20.1`
- Fabric
- Java `17`

Other Minecraft versions and loaders are not silently converted or treated as
compatible. An explicitly unsupported target is rejected before generation or
building; choose `1.20.1` with Fabric or wait for that target to be added.

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
