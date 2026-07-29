# M.M.M Make Minecraft Mode

[English](README.md) | **한국어**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

M.M.M은 Minecraft Java Fabric `1.20.1`용 역할 분리형 제작 시스템입니다.
사용자가 승인한 자연어 요구를 소스와 리소스로 만들고, 정적 검증·Gradle·GameTest·
JAR 검사를 거친 뒤 실제로 통과한 결과만 배포 ZIP에 넣습니다.

## 역할별 모델

모델 ID의 유일한 원본은 `config/model_registry.yaml`입니다.

| 역할 | T4 로컬 모델 |
|---|---|
| 게임 기획·월드 기획·화면 검사 | `Qwen/Qwen3.5-4B` |
| Minecraft 코드 생성 | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| 조사·메모리 안전 코딩 | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| 코드 검색 임베딩 | `Qwen/Qwen3-Embedding-0.6B` |
| 검색 재정렬 | `Qwen/Qwen3-Reranker-0.6B` |
| 콘셉트 이미지 | `black-forest-labs/FLUX.2-klein-4B` |
| 음성 인식 | `openai/whisper-small` |

모델 실패는 `ModelBackendError`로 그대로 반환합니다. 몰래 휴리스틱 결과로
교체하지 않습니다. `HeuristicPlanner`는 명시적으로 선택하는 진단·테스트 용도만
남아 있습니다.

## MCP와 도구 경계

실제 stdio MCP 서버를 실행합니다.

```bash
python -m minecraft_mod_ai.mcp_server
```

`.mcp.json`에는 자체 서버, Minecraft 소스·매핑 MCP, Playwright 조사가 들어
있습니다. `config/external_mcp_registry.yaml`에는 GitHub, JDT Language Server,
Blockbench, disposable Fabric `1.20.1` 런타임, Mineflayer `1.20.1`의 실행 조건과
허용 도구가 기록돼 있습니다.

외부 실행 파일·서버가 없거나 버전이 다르거나, EULA를 승인하지 않았거나, 경로가
작업공간을 벗어나거나, 허용하지 않은 명령이면 성공을 흉내 내지 않고 실패합니다.

## 구현된 제작 모듈

- Fabric `1.20.1` 소스·리소스·Datagen·검증·Gradle·GameTest·JAR·release 패키징
- 버전·loader·mapping·라이선스 metadata를 강제하는 프로젝트 코드 RAG
- Qwen3 Embedding 및 Reranker 선택 실행
- JDT Language Server 진단·workspace symbol adapter
- 제한된 Blockbench MCP allowlist
- GeckoLib `4.8.2` 엔티티 소스·리소스 생성 기반
- 실제 gzip structure NBT·template pool·data-pack/world ZIP을 만드는 월드 컴파일러
- 퀘스트·직업/스킬·경제/상점·GUI/네트워크·파티/길드 typed contract 생성기
- EULA 명시 승인이 필요한 disposable Minecraft 런타임 관리자
- 접속·상태·이동·상호작용·인벤토리 검사용 Mineflayer `1.20.1` 브리지
- 빌드 검증 trace·보상 계산·LLaMA-Factory QLoRA 설정
- `skills/`의 실행 Skill 17개

기반 코드가 생성됐다는 이유만으로 완성 기능이라고 표시하지 않습니다. 해당
플러그인의 JDT·Gradle·GameTest·클라이언트·멀티플레이 gate가 통과해야
runtime-complete가 됩니다.

## Colab

노트북에서 프롬프트와 모델 프로필을 정하고 모두 실행합니다. 노트북은 선택한
GitHub ref를 clone하고, 실제 모델 레지스트리를 출력한 뒤 계획을 만듭니다.
`APPROVE_PLAN=True`가 명시된 경우에만 생성과 빌드를 실행합니다.

직접 모델 ID를 입력하는 칸과 silent fallback은 없습니다.

## Python API

```python
from minecraft_mod_ai import ModAISession

session = ModAISession.with_local_model(
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
    profile="t4_local",
)
reply = session.plan("서리 아이템 2개와 블록 2개, 41x41 아레나를 만들어줘.")
print(reply.message)

if reply.ready_to_build:
    result = session.build(reply, source_only=False)
    print(result.release_zip)
```

## 설치와 테스트

```bash
python -m pip install -e ".[dev,ui]"
python -m compileall -q minecraft_mod_ai tools mcp_gateway.py download_resources.py
python tools/build_colab_notebook.py --check
python -m pytest
```

로컬 모델과 제작 extra:

```bash
python -m pip install -e ".[local-model,rag,image,speech,training]"
./download_models.sh t4_local
```

## 정확한 지원 환경

실행 대상은 Minecraft Java `1.20.1`, Fabric, Java `17`, Yarn
`1.20.1+build.1`입니다. 별도 검토 프로필이 없는 다른 버전·loader·mapping은
거부합니다.

일반 CPU CI는 계약·파서·MCP handshake·정적 생성기·소스 검증을 증명합니다.
실제 T4 모델, Blockbench, Minecraft 서버·클라이언트, Mineflayer는 수동
self-hosted 통합 workflow와 그 artifact가 있어야 검증된 것으로 봅니다.
workflow가 실행되지 않았거나 대기 중인 상태는 성공 증거가 아닙니다.

## 문서

- [전체 제작 구조](docs/PRODUCTION_STACK.md)
- [MCP 보안](docs/MCP_SECURITY.md)
- [파인튜닝 파이프라인](training/README.md)
- [AI·MCP 역할표](docs/AI_MCP_MATRIX.md)

## 라이선스

저장소 코드는 MIT입니다. 외부 프로그램과 모델 가중치는 각각의 라이선스를
유지하며 별도 설치 구성요소로 호출합니다.
