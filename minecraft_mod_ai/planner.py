from __future__ import annotations

import gc
import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .capabilities import capability_manifest_hash
from .knowledge import evidence_for_mvp, evidence_snapshot_hash
from .spec import (
    ArenaSpec,
    BossSpec,
    ContentKind,
    ContentSpec,
    DeferredRequest,
    ModSpec,
    PlatformLock,
    Proposal,
    ProposalStatus,
    SpecValidationError,
)


class Planner(Protocol):
    def plan(self, prompt: str) -> Proposal: ...


THEMES: tuple[tuple[tuple[str, ...], str, str, str, str], ...] = (
    (("달빛", "moon", "lunar"), "moon", "Moonlight", "달빛", "#a6adc8"),
    (("얼음", "서리", "ice", "frost"), "frost", "Frost", "서리", "#89dceb"),
    (("불", "화염", "fire", "flame"), "ember", "Ember", "불꽃", "#fab387"),
    (("번개", "전기", "lightning", "storm"), "storm", "Storm", "폭풍", "#f9e2af"),
    (("그림자", "어둠", "shadow", "dark"), "shadow", "Shadow", "그림자", "#585b70"),
    (("숲", "자연", "forest", "nature"), "grove", "Grove", "숲", "#a6e3a1"),
    (("바다", "물", "ocean", "water"), "tide", "Tide", "파도", "#74c7ec"),
)

ADVANCED_CAPABILITIES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("gui", "화면", "메뉴"), "custom_gui", "phase-3"),
    (("음성", "voice", "tts", "asr"), "voice", "phase-5"),
    (("animation", "애니메이션"), "custom_animation", "phase-5"),
    (("차원", "dimension"), "custom_dimension", "phase-5"),
)


def _contains(prompt: str, words: tuple[str, ...]) -> bool:
    lowered = prompt.lower()
    return any(word.lower() in lowered for word in words)


def _theme(prompt: str) -> tuple[str, str, str, str]:
    for words, stem, english, korean, color in THEMES:
        if _contains(prompt, words):
            return stem, english, korean, color
    return "crafted", "Crafted", "제작", "#cba6f7"


def _count(prompt: str, korean_noun: str, english_noun: str, default: int) -> int:
    patterns = (
        rf"{re.escape(korean_noun)}\s*(\d+)\s*개",
        rf"(\d+)\s*개의?\s*{re.escape(korean_noun)}",
        rf"(\d+)\s*{re.escape(english_noun)}s?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return max(0, min(8, int(match.group(1))))
    return default


class HeuristicPlanner:
    """Offline planner that maps common requests into a strict, safe MVP ModSpec.

    It supports one bounded boss/entity, arena-function, and Blockbench/OBJ
    archetype. Capabilities outside that reviewed slice remain explicit deferrals.
    """

    def plan(self, prompt: str) -> Proposal:
        prompt = prompt.strip()
        if not prompt:
            raise SpecValidationError("프롬프트를 입력해 주세요.")

        stem, english_theme, korean_theme, color = _theme(prompt)
        asks_item = _contains(prompt, ("아이템", "item", "결정", "crystal", "도구", "tool"))
        asks_block = _contains(prompt, ("블록", "block", "광석", "ore", "벽돌", "brick"))
        if not asks_item and not asks_block:
            asks_item = True
            asks_block = True

        item_count = _count(prompt, "아이템", "item", 1 if asks_item else 0)
        block_count = _count(prompt, "블록", "block", 1 if asks_block else 0)
        contents: list[ContentSpec] = []

        item_suffixes = ("crystal", "shard", "relic", "gem", "dust", "core", "token", "charm")
        item_english = ("Crystal", "Shard", "Relic", "Gem", "Dust", "Core", "Token", "Charm")
        item_korean = ("결정", "조각", "유물", "보석", "가루", "핵", "토큰", "부적")
        for index in range(item_count):
            contents.append(
                ContentSpec(
                    content_id=f"{stem}_{item_suffixes[index]}",
                    kind=ContentKind.ITEM,
                    display_name_en=f"{english_theme} {item_english[index]}",
                    display_name_ko=f"{korean_theme} {item_korean[index]}",
                    color=color,
                )
            )

        block_base = "ore" if _contains(prompt, ("광석", "ore")) else "crystal_block"
        block_suffixes = (block_base, "bricks", "tiles", "pillar", "lamp", "stone", "glass", "slab")
        block_english = ("Ore" if block_base == "ore" else "Crystal Block", "Bricks", "Tiles", "Pillar", "Lamp", "Stone", "Glass", "Slab")
        block_korean = ("광석" if block_base == "ore" else "결정 블록", "벽돌", "타일", "기둥", "등불", "돌", "유리", "반 블록")
        for index in range(block_count):
            contents.append(
                ContentSpec(
                    content_id=f"{stem}_{block_suffixes[index]}",
                    kind=ContentKind.BLOCK,
                    display_name_en=f"{english_theme} {block_english[index]}",
                    display_name_ko=f"{korean_theme} {block_korean[index]}",
                    color=color,
                )
            )

        asks_boss = _contains(prompt, ("보스", "boss", "몹", "mob", "entity", "엔티티"))
        asks_arena = _contains(prompt, ("던전", "dungeon", "맵", "map", "world", "월드", "아레나", "arena"))
        asks_3d = _contains(prompt, ("3d", "모델링", "model", "모델"))
        if asks_arena or asks_3d:
            asks_boss = True
        boss = (
            BossSpec(
                entity_id=f"{stem}_warden",
                display_name_en=f"{english_theme} Warden",
                display_name_ko=f"{korean_theme} 수호자",
                primary_color=color,
                secondary_color="#cba6f7",
            )
            if asks_boss
            else None
        )
        arena = (
            ArenaSpec(
                arena_id=f"{stem}_arena",
                display_name_en=f"{english_theme} Arena",
                display_name_ko=f"{korean_theme} 투기장",
                floor_block=(
                    "minecraft:packed_ice"
                    if stem == "frost"
                    else "minecraft:deepslate_tiles"
                ),
                accent_block=(
                    "minecraft:blue_ice"
                    if stem == "frost"
                    else "minecraft:amethyst_block"
                ),
            )
            if asks_arena
            else None
        )

        deferred: list[DeferredRequest] = []
        for keywords, capability, phase in ADVANCED_CAPABILITIES:
            if _contains(prompt, keywords):
                deferred.append(
                    DeferredRequest(
                        capability=capability,
                        reason=(
                            "현재 검증된 MVP archetype은 item/block, biped boss/3D, "
                            "명시 실행형 arena입니다. "
                            "컴파일되지 않은 고급 기능을 완료로 표시하지 않습니다."
                        ),
                        suggested_phase=phase,
                    )
                )

        mod_id = f"{stem}_works"
        spec = ModSpec(
            mod_id=mod_id,
            mod_name=f"{english_theme} Works",
            package_name=f"ai.minecraft.generated.{mod_id}",
            version="1.0.0",
            summary=(
                f"{prompt} 요청에서 검증 가능한 아이템·블록"
                + ("·보스·3D" if boss else "")
                + ("·아레나 맵" if arena else "")
                + " 수직 슬라이스"
            ),
            contents=tuple(contents),
            boss=boss,
            arena=arena,
            platform=PlatformLock(),
        )
        evidence_query = (
            f"{prompt} project build metadata dependency "
            + ("boss entity gametest " if boss else "item block recipe data generation ")
            + ("arena structure runtime" if arena else "")
        )
        evidence_sources = evidence_for_mvp(evidence_query)
        proposal = Proposal(
            schema_version="minecraft-mod-ai/proposal-v1",
            proposal_version=1,
            status=ProposalStatus.AWAITING_APPROVAL,
            requested_prompt=prompt,
            spec=spec,
            assumptions=(
                "Minecraft Java Edition 1.20.1, Fabric, Java 17만 지원합니다.",
                "텍스처는 라이선스 문제가 없는 결정론적 16x16 PNG를 생성합니다.",
                "사용자의 실제 Minecraft 월드에는 쓰지 않습니다.",
            ),
            exclusions=(
                "자동 게시, 실제 사용자 월드 수정, 외부 서버 명령",
                "자유형 애니메이션·임의 차원·GUI·음성 생성",
                "사용자 승인 없이 아레나 함수를 실행하거나 보스를 소환하는 동작",
            ),
            deferred_requests=tuple(deferred),
            acceptance_tests=(
                "승인 해시가 일치해야만 프로젝트 파일을 생성한다.",
                "모든 JSON/PNG/리소스 참조와 ID가 결정론적 검증을 통과한다.",
                "Gradle clean build가 exit code 0으로 끝난 경우에만 JAR를 제공한다.",
                "JAR가 ZIP 형식이며 fabric.mod.json, 클래스, 생성 리소스를 포함한다.",
                "영어와 한국어 번역 키가 모든 생성 콘텐츠를 포함한다.",
                "보스 요청 시 서버 권위 엔티티·보스바·loot·spawn egg와 GameTest를 포함한다.",
                "맵 요청 시 결정론적 arena 함수·WorldDesignIR·경로 검증을 포함한다.",
                "3D 요청 시 Blockbench bbmodel 원본·64x64 texture·런타임 renderer를 포함한다.",
            ),
            evidence_sources=evidence_sources,
            evidence_snapshot_hash=evidence_snapshot_hash(evidence_sources),
            capability_manifest_hash=capability_manifest_hash(),
            imported_source_snapshot_hash="",
            risk_approvals=(
                "Gradle 빌드는 승인 후 공식 Gradle/Fabric/Mojang 저장소에 네트워크로 접근합니다.",
            ),
        ).with_hash()
        proposal.validate()
        return proposal


class LocalTransformersPlanner:
    """Lazy, constrained proposal worker for a Colab T4.

    This optional general instruction model is not a central game-design
    authority. It is loaded only to draft a small ModSpec candidate, constrained
    to the schema, then fully released. The deterministic fallback is always
    disclosed by the ``last_backend`` attribute.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        fallback: Planner | None = None,
        max_new_tokens: int = 1400,
    ) -> None:
        self.model_id = model_id
        self.fallback = fallback or HeuristicPlanner()
        self.max_new_tokens = max_new_tokens
        self.last_backend = ""

    def plan(self, prompt: str) -> Proposal:
        try:
            proposal = self._plan_with_model(prompt)
            self.last_backend = f"local-transformers:{self.model_id}"
            return proposal
        except Exception:
            self.last_backend = "deterministic-fallback"
            return self.fallback.plan(prompt)

    def _plan_with_model(self, prompt: str) -> Proposal:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local model extras are not installed. Install .[local-model]."
            ) from exc

        tokenizer = None
        model = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            schema_prompt = _planner_system_prompt()
            messages = [
                {"role": "system", "content": schema_prompt},
                {"role": "user", "content": prompt},
            ]
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(rendered, return_tensors="pt")
            model_device = next(model.parameters()).device
            inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output[0][inputs["input_ids"].shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            model_data = _extract_json_object(text)
            proposal = _proposal_from_model_data(prompt, model_data)
            proposal.validate()
            return proposal
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SpecValidationError("The local model did not return a JSON object.")


def _planner_system_prompt() -> str:
    return """
You are a Minecraft Fabric 1.20.1 requirements planner. Output exactly one JSON
object and no markdown. You may plan only simple items and blocks. Never emit Java,
paths, shell commands, entities, GUI, worldgen, networking, or publication actions.

JSON contract:
{
  "mod_id": "lowercase_snake_case",
  "mod_name": "English name",
  "package_name": "ai.minecraft.generated.lowercase_name",
  "summary": "short Korean or English summary",
  "contents": [
    {
      "content_id": "lowercase_snake_case",
      "kind": "item or block",
      "display_name_en": "English",
      "display_name_ko": "Korean",
      "color": "#RRGGBB",
      "recipe": true
    }
  ],
  "deferred_capabilities": ["unsupported capability names"]
}
Create 1-8 content entries and preserve the user's theme.
""".strip()


def _proposal_from_model_data(prompt: str, data: dict[str, Any]) -> Proposal:
    exact_keys = {
        "mod_id",
        "mod_name",
        "package_name",
        "summary",
        "contents",
        "deferred_capabilities",
    }
    if set(data) != exact_keys:
        raise SpecValidationError(
            f"Local model returned unexpected fields: {sorted(set(data) ^ exact_keys)}"
        )
    contents = tuple(
        ContentSpec(
            content_id=item["content_id"],
            kind=ContentKind(item["kind"]),
            display_name_en=item["display_name_en"],
            display_name_ko=item["display_name_ko"],
            color=item["color"],
            recipe=item["recipe"],
        )
        for item in data["contents"]
    )
    heuristic_spec = HeuristicPlanner().plan(prompt).spec
    spec = ModSpec(
        mod_id=data["mod_id"],
        mod_name=data["mod_name"],
        package_name=data["package_name"],
        version="1.0.0",
        summary=data["summary"],
        contents=contents,
        boss=heuristic_spec.boss,
        arena=heuristic_spec.arena,
        platform=PlatformLock(),
    )
    deferred = tuple(
        DeferredRequest(
            capability=str(capability),
            reason="고급 capability는 승인된 후속 archetype과 runtime test가 필요합니다.",
            suggested_phase="future",
        )
        for capability in data["deferred_capabilities"]
    )
    base = HeuristicPlanner().plan(prompt)
    proposal = replace(base, spec=spec, deferred_requests=deferred, approval_hash="")
    return proposal.with_hash()
