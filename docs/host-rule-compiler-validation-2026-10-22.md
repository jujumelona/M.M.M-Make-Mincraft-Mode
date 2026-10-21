# 작은 모델용 관찰 규칙 컴파일러 변경·검증 기록

기준 커밋: `0dd57e57200e` · 구현 커밋: `9eb5945ef`, `0e10c0d5d`, `43e608075`

## 확인한 구조적 원인

첨부 실행에는 모델 요청 404개가 남아 있다. 이벤트 선택 단계에서는 약 8,500토큰의
전체 목록을 보낸 뒤에도 `UNBOUND`가 나왔고, 후속 선택에서는 실제로 존재하는
`ServerTickEvents`까지 거절했다. 계절 같은 게임 규칙을 Fabric 이벤트 하나의
이름에 대응시키는 판단이 작은 모델에 남아 있었다. Java를 여러 조각으로 나누어도
이 판단과 조건식·API 조합·실행 순서 작성은 그대로였다.

같은 실패 입력으로 관찰 선택 메시지를 재구성하면 기존 **35,400자 → 897자**다.
기존 실행의 실제 입력 토큰은 8,448개였다. 새 요청은 월드 상태·서버 상태·네이티브
이벤트 중 하나를 고른다. 이는 메시지 구성 비교이며 새 요청의 실제 모델 추론이나
지연 시간을 측정한 결과는 아니다.

## 실제 변경

- 운영 경로 `compose_host_java_source`의 요청 기능 진입점을 새 규칙 컴파일러로
  연결했다. 선택형 실험 플래그를 켜야 사용하는 경로가 아니다.
- 세계·서버의 상태 관찰과 실제 네이티브 이벤트 발생을 구분한다. 상태 관찰은
  호스트가 검사된 틱 ABI에 연결하며, 네이티브 이벤트는 12개씩 확인한다.
  실패한 네이티브 이벤트를 임의의 틱으로 대체하지 않는다.
- 모델 출력은 기존 값 ID, 검사된 API ID, 연산자, JSON 리터럴이다.
  호스트가 인자형·오버로드·범위·선언·Java 문법을 만든다.
- 반복 관찰, 조건이 거짓에서 참으로 바뀌는 순간, 관찰 값 변경을 구분한다.
  처음 관찰 시 실행 여부를 명시하며 월드·서버·엔티티별 이전 값을 분리한다.
- 계산 값과 생성 객체는 호출과 범위별로 한 번만 지연 평가한다. 같은 객체를
  재사용할 때 다시 만들지 않고, 실행하지 않는 조건의 오른쪽 항도 평가하지 않는다.
- 기존 설계의 상수는 원본 데이터 위치와 해시를 보존해 호스트가 복사한다.
  상태 쓰기 권한에는 그 상태를 읽고 갱신할 권한도 포함한다.
- 규칙으로 만든 Java가 컴파일에 실패하면 원본 규칙과 진단을 보존한다.
  기존 자유 형식 Java 수정기로 우회하지 않으며 성공 체크포인트도 만들지 않는다.

특정 계절·우주·상점 프롬프트 전용 생성 템플릿은 추가하지 않았다.

## 검증 결과

새 경로와 직접 관련된 다음 검사는 **81개 통과**했다. 생성한 Java를 실제
`javac`로 컴파일하고 `java`로 실행한 검사도 포함한다.

```bash
python -m pytest \
  tests/test_host_rule_observation.py tests/test_host_rule_program.py \
  tests/test_host_rule_feature.py tests/test_construction_host_source.py \
  tests/test_construction_host_member_repair.py tests/test_host_typed_state_access.py \
  tests/test_integrity_pipeline.py tests/test_host_runtime_integrity_drift.py
```

확인한 실행 사례는 실제 실패 로그의 26.2 틱 ABI, 시계 값 계산, 구간 변경 시
한 번만 실행, 다른 월드의 기록 분리, 거짓 조건에서 실행 차단, 처음 관찰 정책,
콜백 반환, 객체 재사용, 상태 읽기·갱신, 지연 평가 및 자료형·주입 거부다.
변경 파일의 린트, Python 컴파일, `git diff --check`도 통과했다.

전체 검사는 **4,332개 통과, 19개 실패, 7개 건너뜀**으로 끝났다. 기존 커밋을
별도 디렉터리에 그대로 풀어 재검사한
결과, 기존 가짜 모델이 앞선 커밋의 전역 이벤트 선택 요청을 처리하지 못하는
테스트 12개가 수정 전에도 실패했다. 해당 테스트는 다음 파일에 속한다.

- `tests/test_host_callback_runtime.py` — 4개
- `tests/test_host_fabric_callback_abi.py` — 4개
- `tests/test_host_behavior_event_dispatch.py` — 4개

JDT/Gradle 의존성 조회 2개, SOCKS 지원 패키지 미설치 2개, Java 17로 Java 21용
검사를 실행한 항목 1개도 기존 커밋에서 재현했다. 마지막 환경 문제 3개는 해당
테스트 환경을 보완한 뒤 별도로 재검사해 **3개 모두 통과**했다. 공개 코드의
의존성은 바꾸지 않았다.

전체 실행에서 실패한 외부 MCP 범위 캐시와 RAG CPU 설정 검사 2개는 현재 코드와
기존 코드의 단독 검사에서 모두 통과했다. 전체 실행의 순서·환경에 따른 원인은
확정하지 않았다. 전체 결과를 녹색으로 표시하지 않는다.

실패 19개의 정확한 이름과 기준 커밋 재현·추가 검사 결과는 첨부 파일의
`test-results.json`에 보존한다.

## 검증 범위

실제 MiMo 9B 추론 서버로 전체 생성 작업을 다시 실행하지 않았다. Java 실행
검사의 Minecraft/Fabric 쪽은 검사 대상 ABI를 표현한 시험 구현이다.
실제 26.2 의존성 전체를 사용한 빌드와 Minecraft 플레이테스트의 성공을 의미하지
않는다. 원문과 조건식·API 선택의 의미가 일치하는지는 아직 모델이 맡는 부분이며,
실행 기록에도 `model_authored_unverified`, `runtime_verified=false`를 유지한다.

## 조사한 공식 자료와 선택

- [Fabric 이벤트 문서](https://docs.fabricmc.net/develop/events/) — 이벤트와
  사용자 정의 동작의 구분, 검사된 콜백 연결의 근거.
- [Palantir JavaPoet](https://github.com/palantir/javapoet) 및
  [JavaParser](https://github.com/javaparser/javaparser) — Java 생성·분석 도구이며
  게임 동작을 나누는 작업까지 해결하지 않는다. 기존 Tree-sitter·JavaOperations·
  javac를 재사용했다.
- [CEL Java](https://github.com/cel-expr/cel-java) — 자료형을 검사하는 조건식과
  실행을 분리하는 구조를 참고했다. 별도 CEL 실행기를 추가한 것은 아니다.
- [llama.cpp 출력 문법](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
  — 기존 구조화 출력은 유지하되, 출력 문법만으로 게임 의미를 보장한다고
  취급하지 않았다.

## 반영 상태

로컬 `main`에 구현 커밋 3개를 만들었다. 공개 GitHub `main` 푸시는 자동 승인
검토가 새 코드·설계문서의 공개 게시 승인을 요구하며 거부했다. 다른 전송 방식으로
우회하지 않았다. 공개 승인을 받기 전까지 원격 저장소 반영 완료로 표시하지 않는다.
