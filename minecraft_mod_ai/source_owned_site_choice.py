"""Choose an actual inspected Java occurrence site, never a fabricated event.

The shortlist is an evidence filter, not an automatic assignment. The model
must select the exact inspected symbol or explicitly decline. A selected
method still requires compilation and a gameplay acceptance test.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import re

from .fixed_template_generation import _validate_native_arguments
from .java_operation_ir import parameter_types
from .source_owned_hook import SourceHookUnavailable


_STOPWORDS = frozenset({
    "player", "world", "game", "server", "client", "entity",
    "state", "event", "when", "then", "that", "this", "should",
    "have", "with", "from", "into", "requested", "action",
    "effect", "method", "callback", "mod", "minecraft",
})


def _words(text: object) -> set[str]:
    raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text or ""))
    # Java occurrence methods include short but meaningful verbs such as
    # use, hit, buy and pay. The old four-character minimum dropped them,
    # making an actual inspected producer invisible to the selection step.
    words = {x for x in re.findall(r"[a-z][a-z0-9]{2,}", raw.lower())
             if x not in _STOPWORDS and x not in {"the", "and", "for", "are"}}
    # Preserve exact inspected spellings, but also match common inflections:
    # an intent saying "buys" must not hide a real method called "buy".
    variants = set(words)
    for word in words:
        if len(word) > 3 and word.endswith("s"):
            variants.add(word[:-1])
        if len(word) > 5 and word.endswith("ing"):
            variants.add(word[:-3])
            variants.add(word[:-3] + "e")
    return variants


def shortlist_verified_sites(behavior: Mapping, symbols: Mapping, *, limit=24,
                            side="COMMON") -> list[dict]:
    """Include relevant *existing* Java methods, never an invented source owner."""
    words = _words(" ".join(str(behavior.get(key) or "") for key in (
        "trigger", "intent", "source_quote", "effects")))
    for key in ("runtime_state_reads", "runtime_state_writes"):
        for name in behavior.get(key) or ():
            words.update(_words(name))
    if not words:
        return []
    classes = {row.get("owner") for row in symbols.values()
               if isinstance(row, Mapping) and row.get("kind") == "CLASS"}
    matches = []
    for ref, row in symbols.items():
        if not isinstance(row, Mapping) or row.get("kind") != "METHOD":
            continue
        # Source-only client screens cannot become COMMON server Mixins.
        if str(row.get("side") or "COMMON").upper() == "CLIENT" and str(side).upper() != "CLIENT":
            continue
        owner, name = row.get("owner"), row.get("name")
        if (not isinstance(owner, str) or owner not in classes
                or not isinstance(name, str) or not name
                or name.startswith("<")):
            continue
        flags = int(row.get("access") or 0)
        if row.get("static") or flags & (8 | 256 | 1024 | 4096):
            continue
        method_words = _words(name)
        owner_words = _words(owner.rsplit(".", 1)[-1])
        method_matches = method_words & words
        owner_matches = owner_words & words
        if not method_matches:
            # Owner-name overlap alone must not choose arbitrary getters,
            # lifecycle callbacks or world-tick loops as a producer.
            continue
        # Exact method-name evidence is stronger than an owner coincidence.
        score = len(method_matches) * 4 + len(owner_matches)
        if score < 4:
            continue
        source = str(row.get("source_path") or row.get("source_file") or "")
        if source.startswith(("src/main/java/", "src/client/java/")):
            score += 3
        matches.append((score, str(ref), {
            "ref": str(ref), "owner": owner, "method": name,
            "parameter_types": list(parameter_types(row)),
            "return_type": row.get("return_type"),
            "source_path": source or None,
        }))
    matches.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in matches[:limit]]


def choose_verified_source_site(router, behavior: Mapping, symbols: Mapping, *, side="COMMON") -> str:
    behavior_id = str(behavior.get("behavior_id") or "")
    candidates = shortlist_verified_sites(behavior, symbols, side=side)
    if not candidates:
        raise SourceHookUnavailable(
            behavior_id, reason="no_relevant_inspected_source_site", inspected=0)
    valid = {row["ref"] for row in candidates}
    schema = {"type": "object", "properties": {
        "source_site_ref": {"type": "string", "enum": [*sorted(valid), "SOURCE_REQUIRED"]},
    }, "required": ["source_site_ref"], "additionalProperties": False}
    response = router.generate_structured_data(
        "planner", [
            {"role": "system", "content": (
                "Select only an exact inspected Java method that is the ORIGINAL "
                "occurrence source of the requested gameplay behavior. Its "
                "target instance and input arguments must expose the required "
                "runtime observation. Do not choose a similar-looking method "
                "merely because it compiles. If no candidate is the true "
                "occurrence, select SOURCE_REQUIRED. No Java code.")},
            {"role": "user", "content": json.dumps({
                "behavior": {key: behavior.get(key) for key in (
                    "behavior_id", "trigger", "intent", "effects",
                    "source_quote", "runtime_state_reads", "runtime_state_writes")},
                "inspected_source_sites": candidates,
            }, ensure_ascii=False)},
        ],
        data_name="select_inspected_source_producer_site", parameters=schema,
    )
    selected = _validate_native_arguments(response, schema)["source_site_ref"]
    if selected == "SOURCE_REQUIRED":
        raise SourceHookUnavailable(
            behavior_id, reason="custom_source_site_not_resolved",
            inspected=len(candidates))
    return selected
