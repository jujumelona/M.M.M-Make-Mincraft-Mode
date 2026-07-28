# Minecraft Mod AI

자연어 요청을 검토 가능한 제안서로 바꾸고, 사용자가 그 제안서의 SHA-256을
승인한 뒤에만 Minecraft Java용 Fabric 프로젝트를 생성·빌드·검증하는
로컬/Google Colab 파이프라인입니다.

설계 기준은
`Minecraft_Multimodal_Mod_AI_Architecture_2026_v6_Dual_Deployment_Profiles.pdf`이며,
현재 구현은 넓은 설계 전체를 흉내 내는 시뮬레이터가 아니라 실제로 검증한
하나의 수직 슬라이스입니다.

- GitHub: <https://github.com/jujumelona/minecraft-mod-ai>
- Colab:
  <https://colab.research.google.com/github/jujumelona/minecraft-mod-ai/blob/main/Minecraft_Multimodal_Mod_AI_Architecture_v6.ipynb>

## 현재 구현 범위

| 영역 | 실제 동작 |
|---|---|
| 플랫폼 | Minecraft Java 1.20.1, Java 17, Fabric만 지원 |
| 기본 콘텐츠 | 아이템, 블록, 조합법, loot table, 태그, 영문/한글 번역 |
| 보스 | 서버 권위 hostile entity, 체력/공격력 속성, bossbar, spawn egg, 전용 loot |
| 맵 | 관리자가 명시적으로 실행하는 결정론적 arena 함수와 datapack |
| 맵 검증 | `WorldDesignIR`, 2D 미리보기, 5칸 입구→보스 중심 BFS 경로 증명 |
| 3D | 64×64 엔티티 텍스처, 편집 가능한 Blockbench `.bbmodel`, `.obj/.mtl` |
| 런타임 렌더링 | 검증된 vanilla biped `ZombieEntityModel`에 생성 텍스처와 크기 적용 |
| 기존 모드 입력 | 선택적 source/release ZIP 안전 inventory, 기준 SHA-256, 승인 해시 결합 |
| RAG | 공식 출처 allowlist, 1.20.1 버전 범위, 결정론적 evidence snapshot |
| 도구 경계 | MCP 2025-11-25 형식의 고정 capability manifest + 로컬 default-deny broker |
| 빌드 | Gradle `clean build`, JUnit, 헤드리스 Fabric GameTest, JAR 구조/클래스 검증 |
| 실행 환경 | Windows/Linux 로컬 및 Colab Gradio UI |

3D MVP는 임의 메시나 자유형 애니메이션을 게임 런타임에 자동 연결하지 않습니다.
`.bbmodel`과 OBJ는 편집 가능한 원본이고, 게임 안에서는 안정적으로 컴파일되는
biped renderer를 사용합니다. 맵도 전체 월드 세이브나 자연 월드젠이 아니라
`/function <mod_id>:build_<arena_id>`로 만드는 제한된 전투 아레나입니다.

## 검증된 참조 결과

2026-07-28에 서리 보스·아레나·3D·아이템·블록 요청으로 전체 경로를 실행했습니다.

- 결정론적 프로젝트 검사: `PASS` — 153 checks, 0 findings
- Gradle 8.5 `clean build`: `PASS`
- Fabric GameTest: `PASS` — 1/1 required tests
- GameTest 안에서 arena 함수 12개 명령 실행, 바닥/벽 배치, 레시피 로드,
  보스 속성·서버 spawn·arena summon 확인
- JAR 검사: `PASS` — 29 checks, 0 findings
- 릴리스 JAR SHA-256:
  `9e5a281aae33781c3c5349afede75e191d10c29dff2100f5b4f18e9f06ebc486`

설치용 JAR는 모든 게이트가 통과했을 때만 release bundle의 `binaries/`에
들어갑니다. 이 검증 때 생긴 수백 MB의 Gradle 캐시·게임 런타임·이전
시뮬레이터 산출물은 GitHub에 게시하지 않습니다.

## 로컬 실행

필수 조건은 Python 3.10 이상과 Java 17입니다. 첫 빌드는 승인 후 공식
Gradle/Fabric/Mojang 저장소에서 의존성을 내려받습니다.

```powershell
python -m pip install -e ".[dev,ui]"
python -m pytest
```

계획 단계는 기본적으로 파일을 쓰지 않습니다. `--save`를 지정한 경우에만
제안서 JSON을 저장합니다.

```powershell
python -m minecraft_mod_ai.cli plan `
  "서리 보스, 아레나 맵, 3D 모델, 아이템과 블록을 만들어줘" `
  --save proposal.json

Get-Content proposal.json
$approval = (Get-Content proposal.json -Raw | ConvertFrom-Json).approval_hash

python -m minecraft_mod_ai.cli execute proposal.json `
  --approve $approval `
  --output minecraft-mod-ai-output
```

소스만 보고 싶다면 `--source-only`를 사용할 수 있습니다. 이 경우 상태는
`SOURCE_READY`이고 설치용 JAR는 발행되지 않습니다. `--skip-gametest`도 개발
진단용일 뿐이며, GameTest가 없으므로 `VERIFIED`나 설치용 JAR가 나오지 않습니다.

로컬 UI:

```powershell
python -m minecraft_mod_ai.cli ui --output minecraft-mod-ai-output
```

UI에서도 순서는 `계획 생성 → JSON 검토 → 표시된 해시 재입력 → 실행`입니다.

기존 모드 수정 준비에서는 먼저 읽기 전용 inventory를 확인하고, 계획과 실행에
같은 ZIP을 지정합니다.

```powershell
python -m minecraft_mod_ai.cli inspect-existing existing-mod.zip

python -m minecraft_mod_ai.cli plan `
  "기존 모드에 서리 보스와 아레나를 추가해줘" `
  --existing-zip existing-mod.zip `
  --save revision-proposal.json

$approval = (Get-Content revision-proposal.json -Raw | ConvertFrom-Json).approval_hash
python -m minecraft_mod_ai.cli execute revision-proposal.json `
  --approve $approval `
  --existing-zip existing-mod.zip `
  --output minecraft-mod-ai-output
```

계획 후 ZIP의 어느 파일이라도 달라지면 실행 전에 snapshot mismatch로
중단되며 output 디렉터리도 만들지 않습니다.

## Google Colab

노트북은 엔진 ZIP을 요구하지 않습니다. 실행할 때마다 GitHub `main`을
`clone`하거나 기존 checkout을 `fetch` 후 `pull --ff-only`하고, 실제로 사용한
commit SHA를 화면에 남긴 다음 그 소스를 설치합니다.

입력 모드는 명확히 분리되어 있습니다.

1. **새 모드**: 기본값입니다. 업로드 없이 GitHub 최신 엔진과 프롬프트만
   사용합니다.
2. **기존 모드 수정 준비**: 노트북에서 `PATCH_EXISTING=True`로 명시한 때만
   사용자가 소유하거나 수정 권한이 있는 source/release ZIP 하나를 업로드합니다.
   ZIP은 실행하지 않고 먼저 경로·크기·비밀정보·Fabric metadata·파일별 해시를
   검사합니다.
3. JAR만 있는 입력은 metadata/inventory 대상으로만 처리합니다. 소스가 없으므로
   소스 수정 대상으로 광고하지 않습니다.

검사를 통과한 기존 입력의 snapshot hash는 제안서와 승인 SHA-256에 포함됩니다.
따라서 업로드 파일, 공식 근거 목록 또는 허용 도구 목록이 달라지면 이전 승인을
재사용할 수 없습니다. 현재 수직 슬라이스는 이 기준선을 보존한 **새 revision
candidate**를 만듭니다. 임의의 기존 저장소에 최소 unified diff를 자동 적용하는
기능은 아직 구현되지 않았으며 원본 ZIP을 덮어쓰지 않습니다.

기본 planner는 빠르고 재현 가능한 규칙 기반 구현입니다. 선택적 로컬 모델은
제한된 `ModSpec` JSON 후보를 만드는 worker일 뿐 중앙 기획자나 실행 권한자가
아닙니다. 생성·검증·빌드는 언제나 typed proposal, 재입력 승인 해시, 코드 소유
allowlist를 통과합니다.

## RAG와 MCP 연구 반영

이 저장소는 MCP 서버라고 주장하지 않습니다. 현재 구현은 MCP `2025-11-25`의
도구 schema·위험 annotation 개념을 capability manifest에 반영하고, 실제 권한은
기존 로컬 broker가 독립적으로 판정합니다. 검색 문서와 업로드 README는 모두
`data_only`로 취급해 도구 요청으로 승격하지 않습니다.

Fabric API 사실은 코드 소유 official-source catalog에서 같은 1.20.1 범위의
근거만 선택합니다. RAG 통과는 컴파일·GameTest·JAR 검증을 대신하지 않습니다.
구현된 방어와 후속 Datagen/구조 NBT/ModelIR/최소 패치 과제는
[`docs/RESEARCH_AND_SECURITY.md`](docs/RESEARCH_AND_SECURITY.md)에 정리했습니다.

## 승인 및 릴리스 게이트

1. `plan()`은 메모리 안에서 immutable proposal과 승인 해시를 만듭니다.
2. 다른 해시, 변조된 payload, 미승인 상태는 scaffold 전에 거부됩니다.
3. broker는 닫힌 action enum과 승인 output root 경계를 검사합니다.
4. 생성 소스의 JSON, PNG, ID, 번역, recipe/loot, 보스/맵/3D 참조를 검사합니다.
5. 공식 checksum으로 확인한 Gradle 8.5로 wrapper, clean build를 실행합니다.
6. GameTest XML에 정확한 필수 testcase가 있고 failure/error/skipped가 없어야 합니다.
7. JAR의 Fabric metadata, exact entrypoint 클래스, JVM class magic, 전체 리소스,
   CRC와 안전한 ZIP 경로를 검사합니다.
8. 검증한 candidate JAR SHA를 메모리에 묶고 release 복사 전후 같은지 확인합니다.
9. 릴리스 디렉터리는 sibling staging에서 완성한 뒤 원자적으로 이름을 바꿉니다.

실패한 빌드는 `FAILED` 증거 번들과 로그를 남기되 `binaries/`에 JAR를 넣지 않습니다.

## 주요 경로

```text
minecraft_mod_ai/
  spec.py          typed spec, exact platform lock, approval hash
  planner.py       deterministic planner, optional local Transformers planner
  knowledge.py     version-scoped official evidence catalog and snapshot
  capabilities.py  MCP-aligned fixed tool manifest (not a transport server)
  importer.py      non-executing existing-project ZIP inventory and baseline
  broker.py        default-deny local tool policy
  generator.py     Fabric Java/resources, boss, arena, 3D sources
  validator.py     source and JAR validation
  runner.py        checksum-pinned Gradle, timeout/process-tree handling
  pipeline.py      staged generation, build, evidence and release packaging
  webui.py         two-step Gradio approval UI
tests/             approval/import/RAG/security/generation/release regression tests
tools/             reference build and deterministic Colab notebook helper
```

## 고정 버전과 근거

- Minecraft `1.20.1`
- Java `17`
- Yarn `1.20.1+build.1`
- Fabric Loader `0.16.10`
- Fabric API `0.92.11+1.20.1`
- Fabric Loom `1.5.4`
- Gradle `8.5`

근거:
[Fabric 프로젝트 생성](https://docs.fabricmc.net/develop/getting-started/creating-a-project),
[Fabric 모드 빌드](https://docs.fabricmc.net/develop/getting-started/building-a-mod),
[Fabric API 공식 저장소](https://github.com/FabricMC/fabric-api),
[Gradle 공식 checksum](https://gradle.org/release-checksums/).
