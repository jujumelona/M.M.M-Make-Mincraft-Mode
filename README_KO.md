# M.M.M Make Mincraft Mode

[English](README.md)

[![Google Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M은 자연어 요구를 Minecraft Java on the host-selected executable target 모드 기획으로 만들고, 사용자가 대화로 수정·확정한 뒤 새 모드 프로젝트를 만들거나 기존 소스 프로젝트를 수정합니다. **독립 맵, 월드 세이브, 월드 ZIP, schematic, Litematica 파일, 외부 Builder용 블록 변경 작업은 만들지 않습니다.**

월드 생성은 별도 맵 제품이 아니라 일반적인 모드 기능입니다. 구조물·바이옴·차원·광석·configured feature·placed feature는 요청한 모드에 실제로 필요할 때만 만들고, 결과는 Fabric 소스 프로젝트의 코드와 데이터 리소스 안에 남습니다.

## 모드 제작 방법

모든 요청에는 같은 제작 기반을 적용하고 실제 필요한 기능만 추가합니다.

- Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle, Java 버전을 고정합니다.
- 공통·서버 안전 코드와 클라이언트 렌더링·화면·키 바인딩·모델 등록 코드를 분리합니다.
- 아이템·블록·조합법·전리품·태그·모델·blockstate·언어 파일은 타입이 있는 레지스트리와 데이터 리소스로 만듭니다.
- Fabric 이벤트를 우선 사용하고 공개 API로 구현할 수 없는 경우에만 Mixin이나 access widener를 사용합니다.
- 네트워크와 상태 변경은 서버 권한으로 처리하고 검증·권한·속도 제한·영속 상태 스키마·마이그레이션·재시작 테스트를 둡니다.
- 설정·엔티티·렌더링·GeckoLib 애니메이션·음향·명령어·멀티플레이 시스템·모드 내부 월드젠은 요청된 경우에만 추가합니다.
- 정적 검사, Eclipse JDT LS, Gradle, GameTest, 전용 서버 로딩, JAR 검사, 런타임 검사, SBOM, 출처 증거를 통과한 결과만 릴리스합니다.

전체 방법표는 [docs/MOD_DEVELOPMENT_METHODS.md](docs/MOD_DEVELOPMENT_METHODS.md)에 있습니다.

## Google Colab

저장소의 공식 노트북은 [`M.M.M_Make_Mincraft_Mode_Colab.ipynb`](M.M.M_Make_Mincraft_Mode_Colab.ipynb) **하나**입니다.

첫 셀에서 `RUN_MODE`를 선택합니다.

- **Full** — 새 플랜 생성 → 대화로 반복 수정 → 사용자가 명시적으로 확정 → 제작.
- **Plan** — 새 플랜 생성·수정·확정 후 플랜만 저장하고 제작은 하지 않음.
- **Revise** — 본인이 소유하거나 수정 권한이 있는 기존 source/release ZIP 하나를 업로드 → 수정 플랜 생성·확정 → 기존 프로젝트 수정 제작.
- **Execute** — 저장된 플랜을 불러와 전체 내용을 확인·수정하고 사용자가 명시적으로 제작을 승인한 뒤 실행.

엔진 ZIP은 필요 없습니다. 설치 셀은 공식 GitHub `main`을 clone 또는 fast-forward하고, checkout이 정확히 `origin/main`과 같은지 확인한 뒤 실제 사용한 commit을 출력합니다. 엔진이나 설치 코드가 바뀌기 전부터 Colab 탭을 열어 두었다면 노트북을 다시 열고 런타임을 재시작한 뒤 실행합니다.

현재 체크인된 노트북에서 선택할 수 있는 모델 프로필은 다음과 같습니다.

- `Qwen3.5-9B_6GB`
- `Qwen3.6-35B_23GB`
- `Qwen3.6-27B_18GB`
- `Qwen3.6-27B_14GB`
- `mini_mod`
- `fast_test`

선택형 로컬 CUDA llama-server 셀도 같은 planner 설정을 사용합니다. Google Drive 저장은 기본 활성화되어 있으며, 재실행 시 이미 완료된 작업을 불필요하게 다시 만들지 않고 이어서 처리합니다.

`PERFORMANCE_MODE`의 기본값은 `Auto`입니다. 콜드 실행이나 캐시가 맞지 않는 실행에서는 현재 CPU, 시스템 RAM, GPU 여유를 확인하고 하나·둘·가능한 경우 네 개의 공유 llama-server 슬롯을 실측한 뒤 최소 향상 기준을 통과한 결정론적 후보 중 가장 좋은 구성을 선택합니다. 조건이 정확히 같은 튜닝 캐시는 재사용합니다. `Latency`는 한 번에 한 요청의 응답 속도를, `Throughput`은 서로 독립적인 기획·구현 페이지의 동시 처리를 우선합니다. 슬롯마다 모델을 따로 올리지 않고 하나의 상주 모델을 공유하며, 병렬 결과는 항상 결정적인 순서로 병합·검증합니다. 큰 모드는 한 번의 무제한 응답이 아니라 계속 이어지는 페이지로 처리합니다.

## 로컬 Python

```python
from minecraft_mod_ai import CompleteModAISession, resolve_mod_development_methods

methods = resolve_mod_development_methods(
    "계절 농사와 요리가 있는 모드를 만들어줘."
)
print(methods["method_ids"])

session = CompleteModAISession(output_root="mmm-output")
plan = session.plan("계절 농사와 요리가 있는 모드를 만들어줘.")
print(plan.message)
plan = session.revise("전투는 빼고 겨울 온실을 추가해줘.")
result = session.build(plan, source_only=True)
print(result.release_zip)
```

## Codex 플러그인

선택형 플러그인 묶음은 [`plugins/mmm-minecraft-mod-ai`](plugins/mmm-minecraft-mod-ai)에 있습니다. 대화형 시작 스킬과 단계별 M.M.M MCP 설정을 담고 있습니다. `mmm-generation` 서버에는 모드 제작 표면만 노출되며 독립 맵이나 외부 Builder 도구는 노출되지 않습니다. Colab이나 Python 사용에는 플러그인 설치가 필요하지 않습니다.

## 규모

기능 수, 모듈 수, 전체 모드 범위에 제품이 정한 고정 총량 제한은 없습니다. 큰 기획은 한 프롬프트나 한 파일을 계속 키우는 대신 제한된 작업 단위로 나눠 재개 가능하게 처리합니다. Minecraft·Java 형식, GPU·RAM·디스크, 모델 런타임, 세션 제한은 실제 실행 한계이며 작업 단위의 자원 경계로 다룹니다.

## 라이선스

[MIT](LICENSE)
