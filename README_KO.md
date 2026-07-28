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
3. 마지막 셀에서 열린 화면에 만들 모드를 적습니다.
4. `1. 계획 생성`을 누르고 표시된 승인 해시를 아래 칸에 그대로 붙여 넣습니다.
5. `2. 승인 후 실행`을 누릅니다.
6. 완료되면 화면 아래의 **release ZIP**을 내려받습니다.

예시 요청:

```text
얼음 마법을 쓰는 보스, 전투 아레나 맵, 3D 모델,
결정 아이템과 블록을 만들어줘
```

### 기존 모드를 수정할 때

1. Colab의 `PATCH_EXISTING = False`를 `True`로 바꿉니다.
2. 자신이 수정할 권한을 가진 source/release ZIP 하나를 업로드합니다.
3. 나머지는 새 모드와 같은 순서로 실행합니다.

## 만들 수 있는 것

- 아이템, 블록, 조합법, loot table, 태그, 한글/영문 이름
- 보스, bossbar, spawn egg, 전용 아이템과 loot
- 명령으로 설치하는 전투 아레나 맵과 datapack
- 엔티티 텍스처, Blockbench `.bbmodel`, `.obj/.mtl`
- Fabric 소스 프로젝트, 검증된 JAR, 배포용 release ZIP

고정 환경은 Minecraft `1.20.1`, Java `17`, Fabric입니다.

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
