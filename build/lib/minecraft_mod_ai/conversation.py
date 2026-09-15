from __future__ import annotations


def merge_design_brief(current_brief: str, message: str) -> str:
    """Append a natural-language revision without exposing protocol state."""

    current = current_brief.strip()
    revision = message.strip()
    if not revision:
        raise ValueError("Design message must not be empty.")
    if not current:
        return revision
    return f"{current}\n\nUser revision:\n{revision}"
