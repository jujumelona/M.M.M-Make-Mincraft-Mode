from __future__ import annotations

import hashlib
import json
from typing import Any

_NO_RESPONSE = object()


class CustomModuleGenerationError(RuntimeError):
    """Raised when one custom-module generation transaction cannot proceed safely."""


class AtomicJavaDecisionError(CustomModuleGenerationError):
    """A rejected native response, before it could be rendered as Java.

    An omitted response means no response was received. An explicit JSON null is
    still evidence and must remain distinguishable from missing output.
    """

    def __init__(self, message: str, *, response: Any = _NO_RESPONSE) -> None:
        super().__init__(message)
        self.response_text = (
            json.dumps(response, ensure_ascii=False, sort_keys=True)
            if response is not _NO_RESPONSE else None
        )
        self.response_sha256 = (
            hashlib.sha256(self.response_text.encode("utf-8")).hexdigest()
            if self.response_text is not None else None
        )


__all__ = ["AtomicJavaDecisionError", "CustomModuleGenerationError"]
