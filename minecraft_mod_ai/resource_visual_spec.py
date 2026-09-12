"""The small model's semantic boundary; no resource or provider settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisualSpec:
    role: str
    silhouette: str = ""
    materials: tuple[str, ...] = ()
    motifs: tuple[str, ...] = ()
    palette: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VisualSpec:
        allowed = {"role", "silhouette", "materials", "motifs", "palette"}
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise ValueError(
                "Unsupported visual fields; technical settings belong to HOST/registry."
            )
        for name in ("role", "silhouette"):
            text = value.get(name, "")
            if (
                not isinstance(text, str)
                or len(text) > 1024
                or (name == "role" and not text.strip())
            ):
                raise ValueError(f"Invalid visual {name}.")
        sequences = {}
        for name in ("materials", "motifs"):
            items = value.get(name, ())
            if (
                not isinstance(items, (list, tuple))
                or len(items) > 32
                or any(
                    not isinstance(item, str) or not item.strip() or len(item) > 256
                    for item in items
                )
            ):
                raise ValueError(f"Invalid visual {name}.")
            sequences[name] = tuple(items)
        palette = value.get("palette", {})
        if (
            not isinstance(palette, Mapping)
            or set(palette) - {"primary", "secondary", "accent"}
            or any(
                not isinstance(v, str) or not v.strip() or len(v) > 256
                for v in palette.values()
            )
        ):
            raise ValueError("Invalid visual palette.")
        return cls(
            value["role"].strip(),
            value.get("silhouette", "").strip(),
            **sequences,
            palette=dict(palette),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["materials"] = list(self.materials)
        result["motifs"] = list(self.motifs)
        return result

    def prompt_fragment(self) -> str:
        return ", ".join(
            filter(
                None,
                (
                    self.role,
                    self.silhouette,
                    *self.materials,
                    *self.motifs,
                    *(
                        f"{key} color {self.palette[key]}"
                        for key in sorted(self.palette)
                    ),
                ),
            )
        )


def resolve_visual_spec(value: Any, legacy_description: str = "") -> VisualSpec:
    return VisualSpec.from_dict(
        value if value is not None else {"role": legacy_description}
    )
