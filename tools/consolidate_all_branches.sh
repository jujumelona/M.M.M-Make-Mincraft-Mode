#!/usr/bin/env bash
set -euo pipefail

REPORT="docs/branch-consolidation-report.json"
BRANCHES_FILE="$(mktemp)"
MERGED_FILE="$(mktemp)"
DELETED_FILE="$(mktemp)"
trap 'rm -f "$BRANCHES_FILE" "$MERGED_FILE" "$DELETED_FILE"' EXIT

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'

INITIAL_HEAD="$(git rev-parse HEAD)"
git for-each-ref --format='%(refname:strip=3)' refs/remotes/origin \
  | grep -v -E '^(HEAD|main)$' \
  | LC_ALL=C sort -u > "$BRANCHES_FILE" || true

while IFS= read -r branch; do
  [[ -n "$branch" ]] || continue
  if git merge-base --is-ancestor "origin/$branch" HEAD; then
    echo "Already preserved in main: $branch"
  else
    echo "Merging missing branch into main: $branch"
    git merge --no-ff --no-edit "origin/$branch"
    printf '%s\n' "$branch" >> "$MERGED_FILE"
  fi
done < "$BRANCHES_FILE"

python -m compileall -q minecraft_mod_ai tools mcp_gateway.py download_resources.py
python tools/build_colab_notebook.py --check
pytest -q

MERGED_HEAD="$(git rev-parse HEAD)"
if [[ "$MERGED_HEAD" != "$INITIAL_HEAD" ]]; then
  git push origin HEAD:main
fi

while IFS= read -r branch; do
  [[ -n "$branch" ]] || continue
  if ! git merge-base --is-ancestor "origin/$branch" HEAD; then
    echo "Refusing deletion; branch is not preserved in main: $branch" >&2
    exit 1
  fi
done < "$BRANCHES_FILE"

while IFS= read -r branch; do
  [[ -n "$branch" ]] || continue
  echo "Deleting verified branch: $branch"
  git push origin --delete "$branch"
  printf '%s\n' "$branch" >> "$DELETED_FILE"
done < "$BRANCHES_FILE"

python - "$REPORT" "$INITIAL_HEAD" "$(git rev-parse HEAD)" "$BRANCHES_FILE" "$MERGED_FILE" "$DELETED_FILE" <<'PY'
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

report, initial, final, branches_path, merged_path, deleted_path = sys.argv[1:]

def lines(path: str) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]

payload = {
    "schema_version": "mmm/branch-consolidation-v1",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "target_branch": "main",
    "created_branches": [],
    "main_before": initial,
    "main_after_merge": final,
    "discovered_non_main_branches": lines(branches_path),
    "merged_into_main": lines(merged_path),
    "deleted_after_ancestry_and_test_verification": lines(deleted_path),
    "verification": [
        "git merge-base --is-ancestor for every non-main branch",
        "python compileall",
        "Colab notebook regeneration contract",
        "pytest -q",
    ],
}
Path(report).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

git add "$REPORT"
git commit -m "Record verified consolidation of all branches into main"
git push origin HEAD:main
