"""Compatibility entry point for the M.M.M Make Mincraft Mode CLI.

The implementation lives in :mod:`minecraft_mod_ai`.  This file intentionally
contains no simulated build or validation path.
"""

from minecraft_mod_ai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
