from __future__ import annotations

"""Filesystem copy primitives shared by staging and checkpoint code."""

import os
import shutil

_FICLONE = 1074041865


def reflink_or_copy(source: str, target: str) -> str:
    """Copy one regular file, preferring a copy-on-write reflink when supported."""

    if os.name == "posix":
        try:
            import fcntl

            with open(source, "rb") as src, open(target, "wb") as dst:
                fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
            shutil.copystat(source, target)
            return target
        except (OSError, ImportError):
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
    return shutil.copy2(source, target)


__all__ = ["reflink_or_copy"]
