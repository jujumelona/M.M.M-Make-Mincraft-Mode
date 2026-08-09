# M.M.M Make Mincraft Mode

[English](README.md)

[![Google Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M은 말로 입력한 요구를 Minecraft Java 1.20.1 Fabric 모드 기획으로 정리하고, 대화로 수정한 다음 새 모드 프로젝트를 만들거나 기존 소스 프로젝트를 수정합니다. 독립 맵, 월드 세이브, 월드 ZIP, schematic, Litematica 파일, 외부 Builder용 블록 변경 작업은 만들지 않습니다.

월드 생성은 별도 맵 제품이 아니라 일반적인 모드 기능으로 처리합니다. 구조물·바이옴·차원·광석·configured feature·placed feature는 요청한 모드에 실제로 필요할 때만 만들며, 결과는 Fabric 소스 프로젝트의 코드와 데이터 리소스 안에 들어갑니다.

중앙 에이전트가 요청마다 필요한 시스템·코드·라이브러리·이미지·3D·애니메이션·음향·라이선스·테스트 영역을 다시 분류하고, 정확한 버전의 RAG와 호환 가능한 오픈소스 또는 미디어 후보를 검색합니다. 검색 결과는 원본 라이선스와 변경 불가능한 파일 해시를 확인하기 전에는 자동 설치하거나 복사하지 않습니다.

각 요청은 구현 항목과 게임에서 확인할 테스트까지 연결해 보존합니다. 큰 작업은 관련 품질 항목마다 새 검증 결과가 생길 때까지 이어서 처리하고, 계속할 수 없으면 실제로 부족한 외부 입력이나 실행 환경을 정확히 표시합니다. OpenAlex와 Crossref에서 최신 연구 후보도 찾지만, 논문이나 검색 결과만으로 모드가 완성됐다고 판정하지 않습니다.

AI나 음성이 필요한 요청이면 엔진에 특정 제품을 고정하지 않고 실행 시점의 런타임·모델 목록을 다시 검색합니다. 추론·음성 인식·발화 감지·음성 합성·번역·전송·선택적 음성 적응을 나눈 뒤 정확한 Minecraft·Fabric·Java 경계, 모델 revision, 코드·모델·데이터 라이선스, 하드웨어·지연시간 실측, 개인정보와 실패 대안을 확인합니다. 승인된 계획에는 요청된 실행 기능만 허용하는 크기·시간·동시성 제한과 토큰 인증을 둔 비동기 localhost 브리지 하나가 들어가며, 검증 전 모델을 묶거나 모델 출력이 게임 상태를 직접 바꾸게 하지 않습니다. 음성 적응은 허가된 화자이며 명시적 동의·출처·철회·삭제 조건을 모두 통과한 경우에만 사용합니다.

## 모드 제작 방법

모든 요청에는 공통 제작 기반을 적용하고, 실제로 필요한 방법만 추가합니다.

- Minecraft, Fabric Loader, Fabric API, Yarn, Loom, Gradle, Java 버전을 먼저 고정합니다.
- 공통·서버 안전 코드와 클라이언트 렌더링·화면·키 바인딩·모델 등록 코드를 분리합니다.
- 아이템·블록·조합법·전리품·태그·모델·blockstate·언어 파일은 타입이 있는 레지스트리와 데이터 생성기로 만듭니다.
- Fabric 이벤트를 우선 사용하고, 공개 API로 구현할 수 없는 경우에만 정확한 버전에 고정된 Mixin이나 access widener를 사용합니다.
- 네트워크와 상태 변경은 서버 권한으로 처리하고 패킷 검증·권한·속도 제한·영속 상태 스키마·마이그레이션·재시작 테스트를 둡니다.
- 설정·엔티티·렌더링·GeckoLib 애니메이션·음향·명령어·멀티플레이 시스템·모드 내부 월드젠은 요청된 경우에만 추가합니다.
- 정적 검사, Eclipse JDT LS, Gradle, GameTest, 전용 서버 로딩, JAR 검사, 런타임 검사, SBOM, 출처 증거를 통과한 결과만 릴리스합니다.

전체 방법표는 [docs/MOD_DEVELOPMENT_METHODS.md](docs/MOD_DEVELOPMENT_METHODS.md)에 있습니다.

## Google Colab

1. 위 Colab 버튼을 누르고 GPU 런타임을 선택합니다.
2. `PROMPT`에 만들 내용을 적고 셀을 순서대로 실행합니다.
3. 바꾸고 싶은 내용이 있으면 선택형 수정 셀에 말로 적습니다.
4. **이 계획으로 만들기**를 실행한 뒤 결과를 받습니다.

엔진 ZIP은 필요 없습니다. 설치 셀을 실행할 때마다 GitHub `main`을 새로 받거나 fast-forward로 갱신하고, 실제 사용한 commit을 화면에 표시합니다.

- 새 모드: `PATCH_EXISTING=False` 그대로 사용합니다. 업로드가 없습니다.
- 기존 모드 수정: 본인이 소유하거나 수정 권한이 있는 source/release ZIP을 수정할 때만 `PATCH_EXISTING=True`로 바꾸고 ZIP 하나를 올립니다. 소스와 Gradle 프로젝트가 들어 있어야 합니다.
- JAR만 있는 파일은 조사할 수 있지만, 수정 가능한 소스라고 표시하지 않습니다.

기본 저장 위치는 Google Drive입니다. 같은 `RUN_NAME`으로 다시 실행하면 폴더 중복 오류를 내지 않고 완료한 작업부터 이어갑니다.

## 로컬 모델 또는 원격 API

`MODEL_PROFILE="t4_quality"`가 Colab 기본값이며 9B 기획 모델을 4비트로 실행합니다. 더 가벼운 4B 기획 모델은 `t4_local`을 선택합니다. `remote_quality`로 바꾸고 HTTPS API 주소와 모델 이름을 입력하면 OpenAI 호환 원격 API를 사용합니다. API 키는 노트북에 저장하지 않고 실행할 때 숨김 입력으로 받습니다.

로컬 Python에서는 다음처럼 사용할 수 있습니다.

```python
from minecraft_mod_ai import (
    CompleteModAISession,
    resolve_mod_development_methods,
)

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

선택적으로 쓸 수 있는 플러그인 묶음은 [`plugins/mmm-minecraft-mod-ai`](plugins/mmm-minecraft-mod-ai)에 있습니다. 대화형 시작 스킬과 단계별 M.M.M MCP 서버 설정을 함께 담았습니다. `mmm-generation` 서버에는 모드 제작 도구만 노출되며 독립 맵이나 외부 Builder 도구는 노출되지 않습니다. Colab이나 Python 사용에는 이 플러그인 설치가 필요하지 않습니다.

## 규모

기능 수, 모듈 수, 전체 모드 범위에 제품이 정한 고정 총량 제한은 없습니다. 큰 기획은 여러 페이지·작업 조각·체크포인트로 나뉘며, 한 프롬프트나 한 파일이 계속 커지지 않습니다.

무한한 컴퓨터를 뜻하지는 않습니다. Minecraft·Java 형식, GPU·RAM·디스크, 모델 API, Colab 세션에는 실제 한계가 있습니다. 이 한계는 작업 단위의 안전·자원 경계로 두고, 큰 프로젝트는 더 많은 조각과 세션으로 이어서 처리합니다.

## 라이선스

[MIT](LICENSE)
