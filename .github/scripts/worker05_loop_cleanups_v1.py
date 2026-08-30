from pathlib import Path

# The canonical loop already contains these cleanups. Keep this one-shot workflow step
# idempotent and delete the obsolete staging script instead of rewriting current code.
Path(__file__).unlink(missing_ok=True)
