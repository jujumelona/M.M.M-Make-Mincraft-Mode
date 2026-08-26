from __future__ import annotations

import json
from pathlib import Path


def test_colab_exposes_optional_curseforge_key_without_logging_it():
    notebook = json.loads(Path("M.M.M_Make_Mincraft_Mode_Colab.ipynb").read_text(encoding="utf-8"))
    source = "".join(next(cell for cell in notebook["cells"] if cell.get("id") == "configuration")["source"])
    assert 'CURSEFORGE_API_KEY = "" #@param {type:"string"}' in source
    assert 'os.environ["MMM_CURSEFORGE_API_KEY"] = curseforge_api_key' in source
    assert 'colab_userdata.get("CURSEFORGE_API_KEY")' in source
    assert 'print(curseforge_api_key' not in source
