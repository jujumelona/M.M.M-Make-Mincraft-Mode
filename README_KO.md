# M.M.M Make Mincraft Mode

[English](README.md) | **한국어**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/M.M.M-Make-Mincraft-Mode/blob/main/M.M.M_Make_Mincraft_Mode_Colab.ipynb)

자연어로 Minecraft Java용 Fabric 1.20.1 모드를 만들고, 빌드와 검증이 끝난
JAR 및 release ZIP을 내려받는 도구입니다.

> 새 모드를 만들 때는 ZIP을 올리지 않습니다. ZIP 업로드는 이미 만든 모드를
> 수정할 때만 선택적으로 사용합니다.

## 가장 쉬운 사용법

1. 위의 **Open In Colab** 버튼을 누릅니다.
2. Colab에서 `런타임 → 모두 실행`을 누릅니다.
3. 열린 화면에서 만들고 싶은 모드를 AI에게 말합니다.
4. AI가 말로 정리한 계획을 보고, 바꿀 내용은 계속 대화합니다.
5. 계획이 맞으면 `이대로 만들기`를 누르거나 `진행해`라고 입력합니다.
6. 완료되면 **release ZIP**을 내려받습니다.

사용자가 요청하지 않은 콘텐츠는 자동으로 추가하지 않습니다.
간단한 모드는 짧은 제작 계획으로, 대규모 모드는 핵심 플레이 루프,
월드/레벨 설계, 시스템/콘텐츠, 제작 마일스톤과 검증 기준까지 정리합니다.
규모나 첫 플레이 가능 범위가 불명확하면 AI가 먼저 질문합니다.

예시 요청:

```text
단풍 테마 아이템 2개와 블록 3개, 41×41 아레나를 만들어줘.
보스는 넣지 마.
```

### 기존 모드를 수정할 때

1. Colab의 `PATCH_EXISTING = False`를 `True`로 바꿉니다.
2. 자신이 수정할 권한을 가진 source/release ZIP 하나를 업로드합니다.
3. 나머지는 새 모드와 같은 순서로 실행합니다.

## 노트북은 실행기입니다

실제 모드 계획·생성·검사·빌드 기능은 저장소의 Python 패키지에 있습니다.
제공된 노트북은 GitHub의 현재 패키지를 설치하고 화면을 여는 실행기일 뿐,
별도의 오래된 엔진 사본이 아닙니다.

따라서 새 Google Colab 노트북에서도 바로 설치해 사용할 수 있습니다.

```python
%pip install -q --upgrade "mmm-make-mincraft-mode[ui] @ git+https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode.git@main"

from minecraft_mod_ai.webui import launch

launch(output_root="/content/mmm-output", share=True)
```

설치 셀을 다시 실행하면 GitHub `main`의 현재 버전을 받습니다.

## Python API

`ModAISession`은 노트북 화면 없이 계획, 수정, 빌드를 실행하는 간단한 상태형
API입니다.

```python
from minecraft_mod_ai import ModAISession, supported_minecraft_versions

print(supported_minecraft_versions())  # ('1.20.1',)

session = ModAISession(
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
)

first = session.plan("서리 아이템 하나를 만들어줘.")
print(first.message)

revised = session.revise("서리 블록도 하나 추가해줘.")
print(revised.message)

if revised.ready_to_build:  # revised.buildable도 같은 뜻입니다.
    result = session.build(source_only=False)
    print(result.release_zip)
```

- `plan(요청)`은 새 계획을 시작하고 대화형 응답을 반환합니다.
- `revise(변경 내용)`은 현재 계획을 수정하고 새 응답을 반환합니다.
- `reply.message`, `reply.questions`, `reply.ready_to_build`에서 준비된 내용과
  추가로 정해야 할 내용을 확인할 수 있습니다.
- `build(source_only=False)`는 빌드를 수행하고 release 결과를 반환합니다.
  생성 소스만 필요하면 `source_only=True`를 사용합니다.

### 선택적 OpenAI 호환 외부 API

기본값은 내장 planner입니다. Colab에서 외부 OpenAI 호환 HTTPS
chat-completions API를 사용하려면 비공개 Colab Secret `MMM_API_KEY`를 만들고
노트북 접근을 허용하세요. 키를 노트북 셀, 프롬프트, 저장소 파일 또는 공유
링크에 직접 넣지 마세요.

```python
from google.colab import userdata

from minecraft_mod_ai import ModAISession

api_key = userdata.get("MMM_API_KEY")
if not api_key:
    raise RuntimeError("MMM_API_KEY Colab Secret과 노트북 접근 권한이 필요합니다.")

session = ModAISession.with_openai_compatible_api(
    base_url="https://your-provider.example/v1",
    model="your-model-name",
    api_key=api_key,
    output_root="/content/mmm-output",
    minecraft_version="1.20.1",
)
```

제공된 실행 노트북에서는 `api`를 선택하고 HTTPS API 주소와 모델 이름만
입력합니다. 키는 `MMM_API_KEY` Secret에만 보관합니다.

## 현재 생성 결과

- 아이템, 블록, 조합법, loot table, 태그, 한글/영문 이름
- 명시적으로 요청한 경우에만 만드는 보스 엔티티, spawn egg와 loot
- 명시적으로 요청한 경우에만 만드는 아레나 맵과 datapack
- 엔티티 텍스처, Blockbench `.bbmodel`, `.obj/.mtl`
- Fabric 소스 프로젝트, 검증된 JAR, 배포용 release ZIP

## 지원 Minecraft 환경

현재 지원 대상은 정확히 다음과 같습니다.

- Minecraft Java Edition `1.20.1`
- Fabric
- Java `17`

다른 Minecraft 버전이나 loader를 호환되는 것처럼 자동 변환하지 않습니다.
지원하지 않는 대상을 명시하면 생성이나 빌드 전에 거부합니다. 현재는 Fabric
1.20.1을 선택하거나 해당 대상이 추가될 때까지 기다려야 합니다.

## 출력

release ZIP은 다음과 같이 구성됩니다.

```text
art_sources/   3D 모델과 텍스처
binaries/      검증을 통과한 설치용 JAR
docs/          설치 및 관리자 안내
evidence/      빌드와 검사 결과
packs/         아레나 datapack
source/        생성된 Fabric 소스 ZIP
world/         맵 설계 JSON과 미리보기
```

빌드나 검증에 실패하면 설치용 JAR는 `binaries/`에 넣지 않습니다.

## 로컬에서 실행

Python 3.10 이상과 Java 17이 필요합니다.

```powershell
python -m pip install -e ".[ui]"
python -m minecraft_mod_ai.cli ui
```

테스트:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## 프로젝트 구조

```text
M.M.M_Make_Mincraft_Mode_Colab.ipynb  Colab 실행 노트북
minecraft_mod_ai/                     모드 생성·빌드 프로그램
tests/                                자동 테스트
tools/                                Colab 및 빌드 도구
```

## License

[MIT License](LICENSE)입니다. 상업적 이용, 수정 및 재배포가 가능하며 배포할
때 저작권 고지와 라이선스 문구를 포함해야 합니다. 소프트웨어는 보증 없이
제공되며 정확한 조건은 `LICENSE` 전문을 따릅니다.
