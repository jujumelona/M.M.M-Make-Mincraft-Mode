# 플래너 로그 장애 수정 보고서 — 2026-09-06

대상 저장소: `jujumelona/M.M.M-Make-Mincraft-Mode`, 기준 커밋 `82757b774707cdb6f6e854b31b0cf2577f06ed48`.

첨부 로그 전체를 분석한 결과, 직접 중단은 호스트가 생성한 선행 검증 작업 7개를 생산 작업으로 전달한 데서 발생했다. 설계 필드의 잘못된 형식과 빈 값을 성공으로 덮는 동작, 조사 근거의 출처 구분, 파생 요구사항의 소유권도 함께 수정했다.

원본 로그 SHA-256: `de12c38e32583f4bd073874e587855b7c264341c8b98d8fcf9cb1040e6cc316f`.

## 문제별 처리

| 번호 | 확인한 문제 | 수정·검증 내용 |
| --- | --- | --- |
| F01 | `prerequisite_gate` 7개가 생산·자산 연결 없이 실행 DAG에 들어가 14개 링크 오류 발생 | 가짜 작업 생성과 `requirement_ready`를 삭제했다. 실제 최초 작업이 선행 요구사항의 `requirement_done`을 소비한다. 런타임 판정은 자연어가 아닌 `capability:` 제공 계약을 사용한다. 테스트 전용 런타임 작업의 거부는 유지한다. |
| F02 | 결정적인 실행 연결 오류를 조사 88회, 약 900.8초 이후 발견 | 조사 증강 전에 실제 실행 변환·handoff·전체 linker 검사를 실행한다. 증강 이후에도 다시 검증한다. 호스트의 필수 기능 소유권 누락 역시 모델 호출 전에 검사한다. |
| F03 | 의미 추출에서 `싸움` 누락 및 확장·특수광물 범위의 중복 | 기준 main에 이미 있던 `semantic_source_fidelity` 검사를 유지하고 회귀 테스트를 실행했다. 필드 분리 후에도 승인된 원문 범위·정확한 요구사항 ID 계약을 유지한다. 이 항목은 새 의미 추출 모델의 성공률을 측정한 결과가 아니다. |
| F04 | 축약·발명된 요구사항 ID 때문에 전체 필드를 기본값으로 대체 | 구현 설계는 요구사항 하나씩 생성하며 호스트가 모듈 ID와 요구사항 참조를 배정한다. 기타 필드의 미승인 ID는 정확한 오류와 이전 응답을 포함해 해당 필드만 한 번 재생성한다. 여전히 잘못된 출력은 거부한다. |
| F05 | `# ## modules`, `# ## assets` 제목 파싱 실패. 자산 부분은 실제로 비어 있음 | 내용 손실 없는 제목 정규화를 추가했다. 빈 본문은 실패로 남긴다. 자산 불필요 결정은 요구사항마다 `none: <기존 자산으로 충분한 이유>`가 있어야 승인하고 근거를 기록한다. |
| F06 | 빈 객체·목록, 일반 문장, 가짜 모듈을 넣은 뒤 성공 처리 | `_host_field_fallback`, `_ensure_module_coverage` 및 관련 기본값 합성 코드를 삭제했다. `modules: none`도 구현 의무로 통과하지 못한다. 정상 필드의 응답은 보존하고 실패한 필드만 재생성한다. |
| F07 | 내부 `design_*` ID를 플랫폼 검색어로 사용 | 명시적인 게임 기능을 우선 검색한다. 내부 설계 ID는 검색어에서 제외하고 실제 외부 플러그인 ID는 유지한다. |
| F08 | 작은 모델에 큰 섹션·전역 요구사항·출처·ID·형식을 한꺼번에 요구 | 한 호출이 한 필드만 작성한다. modules/assets/combat/mod_context/acceptance_tests는 요구사항 하나로 범위를 제한하고 관련 외부 근거를 선별한다. 제목 계층을 core-loop 행동으로 잘못 수집하는 문제도 수정했다. 호출당 범위를 줄인 것이며 전체 호출 수나 실제 지연 시간 감소를 주장하지 않는다. |
| F09 | 배열이 JSON 문자열로 이중 인코딩됨. 바깥 예외에서 원문을 잃음 | `evidence_refs`, `implementation_obligations`, `acceptance`에 한해서 JSON 문자열 배열을 손실 없이 해제한 뒤 엄격한 스키마 검증을 한다. 임의 문장·숫자·객체를 배열로 강제 변환하지 않는다. router 예외의 실제 output을 복구하므로 이 형식 오류는 추가 모델 호출 없이 처리된다. |
| F10 | 내부 플랫폼 요약과 출처 ID만 있는 메타데이터가 외부 조사 근거로 승격 | 외부 URL 또는 저장소+커밋 출처와 실질적인 본문/주장이 함께 있는 자료만 수집한다. 중첩된 본문에는 상위 출처를 전달한다. 요구사항 관련성이 없는 근거를 제외하는 기존 규칙도 유지한다. |
| F11 | 같은 상위 요구사항의 모든 작업에 파생 의무를 복사 | 호스트가 해당 기능을 소유하는 작업 하나를 지정하고 `owner_task_ref`를 기록한다. 실행 변환은 존재하는 작업인지, 상위 요구사항과 일치하는지 확인하며 다른 작업에는 의무를 복사하지 않는다. |
| F12 | 구현 작업이 없으면 필수 기능까지 `not_applicable` | 기능 적용 여부를 호스트의 implementation capability 계약에서 판단한다. 필수 기능의 소유자가 없으면 `missing`으로 거부한다. 검증된 retain 결정은 기존 구성요소가 책임지므로 새 작업을 요구하지 않는다. |
| F13 | 오래된 테스트가 가짜 성공·섹션당 한 호출·구형 7-facet 응답을 기대 | 해당 설계·조사 테스트를 새 계약으로 갱신했다. 기록된 작업 DAG 재생, 실제 컴파일→변환→handoff→linker, 호스트 작업 확장→생산 계약 테스트를 추가했다. CI의 pre-design 회귀 작업에도 연결했다. |
| F14 | 단계별 추적 ID 분리, 커밋 정보 없음, 깊이·길이 제한으로 진단 내용 손실 | complete planning 실행 ID를 단계 추적에 연결한다. 시작 시 Git SHA와 tracked 변경 여부를 기록한다. 축약된 root 이벤트에는 비밀 키를 가린 전체 JSON 아티팩트의 경로·SHA-256·크기를 연결한다. 각 필드/요구사항/시도와 실패 원문도 저장한다. |
| 추가 | 생산 계약 경계에서 이미 삭제된 `_requirement_acceptance` 호출로 `AttributeError` | 오래된 마이그레이션 전체를 제거했다. 동결된 계획을 검증하고 승인된 요구사항·작업 ID·acceptance를 재작성하지 않는다. 실제 호스트 작업 확장과 생산 계약 컴파일을 연결해 검증했다. |

## 구조 정리

기존 `agentic_research_game_design.py`는 조정 역할로 줄였다. Markdown 파싱은 `design_markdown.py`, 요구사항 검증은 `design_requirement_contract.py`, 조사 문맥은 `design_research_context.py`, 호스트 필드 정의는 `design_section_schema.py`, 개별 호출과 재생성은 `planner_field_worker.py`가 담당한다. 조사 응답 형식 복구와 전체 추적 아티팩트 저장도 각각 독립 모듈로 분리했다.

실패를 성공으로 바꾸는 기본값, 가짜 모듈 합성, 사용하지 않는 섹션 프롬프트, 선행 검증 작업, 생산 경계의 오래된 acceptance 재작성 코드는 삭제했다. 검증기를 비활성화하거나 실패 테스트를 skip/xfail 처리하지 않았다.

## 근거와 적용 범위

복잡한 문제를 작은 부분으로 분해하는 방향은 [Least-to-Most Prompting 연구](https://arxiv.org/abs/2205.10625)를 참고했다. 본 변경은 이 원칙을 필드와 요구사항의 호스트 소유권에 적용한 설계 판단이며, 해당 논문의 실험 결과를 Qwen 9B의 성능 보장으로 옮겨 해석하지 않는다.

실행 변환 전후의 명시적 계약과 전체 검증은 [MLIR Dialect Conversion의 legality 모델](https://mlir.llvm.org/docs/DialectConversion/)을 참고했다. MMM에 MLIR 의존성을 추가하지 않았다. JSON 배열과 문자열은 서로 다른 타입이므로, 제한적인 손실 없는 디코딩 뒤 [JSON Schema 배열 계약](https://json-schema.org/understanding-json-schema/reference/array)을 그대로 검증한다. Qwen native inference의 문법 제약 설정을 임의로 바꾸지 않았다.

## 검증 결과와 재현

- 관련 25개 테스트 파일: **144 passed** (Python 3.12, 프로젝트의 `mcp==2.0.0`).
- 전체 Python 소스·테스트의 CI lint 명령 `ruff check ... --select F,E7,E9`: 통과.
- 런타임 mutation audit 실행 및 패키지 import preflight: 통과.
- 첨부 로그의 완전한 작업 기록 161개와 컴파일 입력을 `tests/fixtures/planner_failure_2026_09_05.json`에 보존했다. 깊이 제한으로 이미 잘린 진단 전용 필드는 재현 입력에서 제외했다.
- 재컴파일 결과 **154개**, 링크 오류 **0개**. 남은 모든 작업 쌍에 대해 원래 DAG의 선후 관계가 정확히 보존됨을 검사했다.
- 원래 가짜 작업 7개를 다시 넣으면 연결 누락으로 거부된다. 실제 런타임 계약이 테스트만 소유하는 경우도 계속 거부된다.
- 실제 컴파일·handoff·linker 검증기는 mock하지 않았다. 생산 구현 의무의 내용 생성은 통제된 모델 응답 fixture를 사용했다. 별도의 순서 테스트만 실패 지점을 주입해 조사 이전 중단을 검증한다.

핵심 재현 명령:

```bash
python -m pytest -q \
  tests/test_planner_log_replay_2026_09.py \
  tests/test_planner_field_evidence_regressions.py \
  tests/test_research_execution_contract.py \
  tests/test_semantic_source_fidelity.py \
  tests/test_plan_collect_all_linker.py
```

전체 저장소가 모두 green이라는 주장은 하지 않는다. 별도로 실행한 기존 `test_evidence_first_planning.py`의 13개 실패는 변경 전 archive에서도 확인했다. 기존 session integration의 4개 실패도 변경 전 재현했고, 그중 생산 계약의 삭제된 함수 호출은 이번에 해결했다. 나머지 3개는 `weather_compass` 모듈 이름을 승인된 의미 요구사항으로 취급하던 오래된 retain fixture와 현재 원문 기반 catalog 계약이 일치하지 않는 문제다. 이 테스트들은 삭제하거나 비활성화하지 않았다.

실제 Qwen 9B 추론, Colab GPU, Minecraft/Gradle 빌드·GameTest 전체 실행은 이 환경에서 수행하지 않았다. 따라서 모델의 새 의미 추출 성공률, 전체 생성 성공, 실제 지연 시간은 후속 실행에서 확인해야 한다. 이번에 검증한 것은 로그에서 확인된 호스트 계약·파싱·근거·소유권·진단 문제와 그 재발 방지다.
