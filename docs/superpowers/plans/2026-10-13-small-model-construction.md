# M.M.M 작은 AI 모드 제작 구조 설계 및 실행계획

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. 구현은 이 문서 검토 이후 직접 진행한다. 각 작업은 체크박스로 추적한다.

**Goal:** 작은 AI가 감당하는 작업으로 실제 Minecraft 모드를 제작하고, 요청이 많아져도 기능을 보존하면서 재사용·이식·연결·새 구현을 수행한다.

**Architecture:** 원본 요청에서 관찰 가능한 행동을 분해하고, 실제 소스와 대상 API를 연결한 제작 작업 그래프를 만든다. AI는 작업별 알고리즘과 타입이 확인된 연산을 작성하고, 호스트는 범용 Java 구문을 렌더링한다. 기존 Typed PlanIR, 상태 컴파일러, 소스 고정·검증, 작업 그래프와 체크포인트는 재사용한다.

**Tech Stack:** 기존 Python 엔진, Fabric 대상 프로젝트, 현재 배포 모델 MiMo-V2.6-Distill-Qwen-9B, llama.cpp, 기존 Java 파서 및 Gradle 검증.

**Spec:** 이 문서의 ‘설계 계약’ 절. 상태는 **검토용 설계**이며 제품 코드 변경·실제 모델 생성·모드 빌드 완료를 뜻하지 않는다.

## 조사 기준과 확인된 사실

읽은 첨부:
- 붙여넣은 마크다운(20261012-134329).md — 실행 커밋 543201dc611b
- 붙여넣은 마크다운(20261012-144618).md — 실행 커밋 53454c1b4f14
- 붙여넣은 마크다운(20261012-164345).md — 실행 커밋 1128e144bdf8

현재 코드 조사 기준: [a77c0f6efe55b03b735bacc4ccad4a7a1b3ebe9a](https://github.com/jujumelona/M.M.M-Make-Mincraft-Mode/commit/a77c0f6efe55b03b735bacc4ccad4a7a1b3ebe9a). 첨부 실행 이후 Java 문법 생성기 변경이 추가되었지만, 해당 커밋 설명에도 실제 모델 추론과 제작 검증을 실행하지 않았다고 기록되어 있다.

세 실행 모두 planner 요청 209회, coder 요청 43회 뒤 동일한 generate-content-00000002에서 실패한다. coder의 43번째 요청은 submit_java_implementation_1이다. 앞선 42회는 import·단위 계획·선언 메타데이터 요청이다. 이는 첫 구현이 성공했다는 증거가 아니다.

| 실행 | 첫 생성 실패 | 최대 요청 입력 |
|---|---|---:|
| 134329 | JavaRegionParseError, 2:101, private static final | 58,099자 |
| 144618 | JavaRegionParseError, 2:306, 닫는 괄호 | 61,675자 |
| 164345 | 첫 구현 요청의 LlamaSemanticProgressTimeout | 61,646자 |

마지막 실행은 ‘120초 동안 의미 있는 진행 없음’으로 중단되며 호출 총 경과시간은 약 240초다. 120초와 전체 호출시간을 혼동하지 않는다.

실패 작업에 붙은 하나의 요구는 ‘우주선 조립, 거래와 구매, 무기·승무원·성능 업그레이드, 확장, 우주 출발’이다. 현재 request_requirements.validate_leaf_atomicity에 이 문장을 넣으면 COMPOUND로 판정한다. AST로 확인한 기본 request_scope 경로는 이 검사를 호출하지 않는다.

실제 잘못 생성된 Java 원문과 RAG의 후보·탈락·재사용 감사 자료는 첨부 콘솔에 없다. 따라서 어느 원문이 괄호 오류를 만들었는지, 이번 실행에서 재사용이 정확히 몇 건이었는지, 마지막 정지가 모델·문법 샘플러·서버 자원 중 무엇에 의해 발생했는지는 아직 확정하지 않았다. 문법 오류와 정지 현상을 모두 ‘모델의 추론 능력’ 하나로 단정하지 않는다.

## Global Constraints

- 사용자 요청의 기능 수와 의미를 보존한다. 총 기능·작업·클래스 수에 제품 차원의 상한을 넣지 않는다.
- 작은 응답으로 나눌 때 입력과 의사결정 난이도도 함께 줄인다.
- 같은 대형 Java 요청을 재시도하거나 출력 한도·타임아웃만 늘리는 방법을 해결책으로 사용하지 않는다.
- 고정된 게임 기능 목록만 호스트가 생성하도록 바꾸지 않는다. AI가 새 동작과 알고리즘을 작성하는 경로를 유지한다.
- 검색 결과의 존재와 실제 소스 재사용을 구분한다. 검색·검증 실패를 조용히 ‘없음’으로 치환하지 않는다.
- 대상 버전·로더·매핑·JDK를 한 실행 안에서 고정한다. 사용자 지정 버전을 임의로 내리지 않는다.
- 기존 결정적 백엔드 검사를 우회하거나 미지원 기능을 지원한다고 표시하지 않는다.
- 승인된 원본 계획과 성공한 산출물을 보존한다. 실패한 기능을 누락시켜 완료로 처리하지 않는다.
- 저장소 변경은 main에 단계별 커밋하고 검증 결과를 기록한다. 설계를 확인하기 전 제품 코드를 변경하지 않는다.
- Python 테스트 통과를 실제 MiMo 9B 제작 성공으로 확대 해석하지 않는다.

## Review Focus

1. 복합 문장을 하나의 기능으로 반환하거나 너무 일찍 complete를 반환해도 요구를 누락시키지 않는다.
2. 검색 결과 0건, 제공자 오류, 다운로드 오류, 대상 컴파일 탈락을 서로 다른 상태로 남긴다.
3. 같은 이름의 오버로드, 클라이언트 전용 API, 오래된 매핑 이름을 잘못 연결하지 않는다.
4. 작업이 많아져도 전체 상태·이전 완성 코드가 매 호출 입력에 누적되지 않는다.
5. 재개 시 구매·보상·등록이 중복 실행되지 않으며 실패한 노드 때문에 계획이 손실되지 않는다.

## 설계 계약

### 1. 요청은 행동으로 나누고 관계를 보존한다

원본 문장, 요청 기능, 구현 작업을 구분한다. ‘우주선을 조립한 뒤 구매·업그레이드하고 출발한다’는 여러 행동과 순서 제약을 포함한다. 하나의 콜백 클래스에 이 문장 전체를 다시 주지 않는다.

행동마다 원본 요구 ID, 발동 조건, 필요한 상태, 관찰 가능한 결과, 앞뒤 관계를 저장한다. 기존 원자성 검사는 분해가 필요한 항목을 찾는 신호로 재사용한다. 정규식 판정 자체가 행동 명세나 완전한 분해를 대신하지 않는다.

복합·미분해 항목은 부모 요구로 남기고 자식 작업으로 구체화한다. source_recovery 원문 보존은 복구 자료이며, 곧바로 생성 가능한 작업이라는 뜻으로 사용하지 않는다. 요구마다 구현 담당과 실행 결과를 연결한 검증 항목이 있어야 한다.

### 2. RAG는 실제 연결 가능한 코드 자료를 만든다

검색은 필요한 행동별로 하고, 후보 저장소를 찾은 뒤 파일·클래스·메서드·상태·리소스 수준으로 조사한다. 기존 저장소 고정, 파일 해시, 라이선스 확인, 의존 관계 검사를 유지한다.

각 후보는 다음 자료를 갖는다.

- 저장소, 정확한 커밋, 라이선스, 원본 경로와 해시
- 재사용할 클래스·메서드 및 대상에 설치할 경로
- 매개변수·반환형·공통/클라이언트 구분·필요한 외부 의존성
- 초기화·콜백·상태 저장·리소스 연결 지점
- 원본과 대상 API의 차이, 이식 작업 목록
- 구현하는 행동과 아직 구현하지 않은 행동
- 대상 컴파일·동작 검증 자료 및 후보 탈락 이유

한 저장소가 기능 전체를 구현해야만 유용한 것은 아니다. 행동을 구성하는 일부 로직을 실제로 재사용할 수 있다. 반대로 관련 단어가 있다는 이유만으로 재사용에 포함하지 않는다.

현재 source_transplant의 의존 관계 자료를 활용하되 코드 전달은 AST의 완전한 선언·본문 경계로 선택한다. authored_source_rag의 2,400자 잘라내기와 총 5,000자·3조각 선택을 ‘호출 가능한 코드 계약’으로 취급하지 않는다. 코드가 길면 관련 메서드와 호출 관계를 추가 작업으로 나누고 필요한 자료를 별도 페이지로 공급한다.

대상 버전에서 바로 검증된 소스는 설치 작업으로, 이식이 필요한 소스는 정확한 메서드별 이식 작업으로 보낸다. 검색이나 이식이 실패했다고 복합 기능 전체를 새 Java 생성 한 번으로 넘기지 않는다. 후보가 실제로 없는 부분도 작은 구현 작업으로 분해한다.

### 3. 제작 계획에서 재사용·연결·새 구현을 명시한다

작업 계약 v1은 다음 필드를 고정한다.

| 필드 | 의미 |
|---|---|
| task_id / requirement_ids | 작업 및 원본 요구와의 연결 |
| kind | reuse, port, glue, logic, resource, verify 중 실제 수행 방식 |
| intent | 이 작업이 구현하는 한 가지 명확한 결과 |
| depends_on | 필요한 선행 작업 |
| source_refs | 고정된 원본 파일·심볼·검증 자료 |
| target_path / owner_symbol | 설치 또는 생성할 파일·심볼 |
| input_symbols / output_symbols | 정확한 타입과 호출 관계 |
| state_reads / state_writes | 필요한 상태와 변경하는 상태 |
| trigger / effects | 발동 조건과 실제 효과 |
| allowed_writes / expected_old_hashes | 이번 작업이 변경할 범위 |
| verification_ids | 성공을 판정할 검증 |
| status | planned, ready, running, verified, failed, blocked |

재사용된 코드를 누가 호출하는지, 초기화를 누가 등록하는지, 기존 상태와 어떻게 연결하는지까지 별도의 glue 작업으로 작성한다. ‘기존 코드 참고’라는 문구만 넣는 것으로 계획을 끝내지 않는다.

소스 검색 결과가 계획을 바꿀 때는 실행 계획의 새 버전을 만들고 원본 요구와 변경 이유를 보존한다. 기능 삭제로 부족한 구현을 해소하지 않는다.

### 4. 작은 AI의 Java 생성 계약을 바꾼다

현재 구조는 호스트가 패키지·클래스·메서드의 틀을 만들지만 AI가 여전히 Java 표현식·본문을 자유 형식으로 작성한다. 작게 나눈 응답에도 전체 요구·상태·선행 코드가 전달된다. 바깥 중괄호와 선언만 고정하면 이 두 문제가 해결되지 않는다.

제안은 **타입이 확인된 범용 Java 연산 그래프**를 AI가 작업별로 작성하게 하는 것이다. AI는 조건·반복·계산·상태 변경·검증된 API 호출·새 helper 관계를 설계한다. 호스트는 그 의미를 보존하면서 Java 구문을 렌더링한다.

기본 구문에는 리터럴, 변수 선언과 참조, 필드 접근, 호출, 생성자, 대입, 연산자, 조건문, 반환, 반복문, switch, 람다와 블록을 포함한다. 게임별 ‘우주선/무기/거래’ 기능 목록으로 제한하지 않는다.

API와 타입은 대상 클래스패스에서 확인한 심볼 ID로 참조한다. 선언 이름·오버로드·매개변수·반환형은 렌더링 전에 결정한다. 모델이 반환형이나 타입 문자열에 Java 선언 전체를 끼워 넣을 수 없도록 타입 참조도 구조화한다.

한 응답은 깊은 재귀 JSON 전체가 아니라 하나의 연산 노드 또는 작은 선형 묶음이다. 의존하는 값과 심볼만 입력에 넣는다. 전체 클래스·모든 helper·전체 상태 모델을 다시 보내지 않는다. 노드 수가 늘어나면 저장된 그래프에 이어 작성한다. 총 알고리즘 크기는 제한하지 않는다.

원시 Java가 반드시 필요한 언어 기능은 명시적인 지원 확장 항목으로 기록하고 해당 렌더러를 구현·검증한 다음 사용한다. 미지원 구문에서 기존 자유 Java 생성기로 자동 복귀하지 않는다.

이 구조가 작은 모델의 의미적 정확성을 보장한다고 주장하지 않는다. 먼저 실제 MiMo 9B로 작은 상태 계산·실제 Fabric 연결·소스 이식의 성공을 확인하고, 같은 실패 요청까지 확장해야 한다.

### 5. 문법 생성기의 정확성과 실행 정지도 별도로 검증한다

최근 Java 문법 생성기 변경은 부가 수정일 뿐 새 제작 계약의 검증을 대신하지 않는다. 실패 Java 원문, 사용 문법의 식별값, 마지막 실제 출력, 서버 상태와 자원 상태를 확보해야 정지 원인을 구분할 수 있다.

기본 제작 경로는 작은 노드용 JSON 제약과 호스트 Java 렌더링을 사용한다. 광범위한 Java 문법을 작은 모델에 매번 샘플링시키는 경로를 기본 해결책으로 유지하지 않는다.

기존 120초 무진행 감시는 유지한다. 진단 없이 시간을 늘리지 않는다. 모델 출력 완료, 구조 검증, 대상 컴파일, 기능 동작 성공을 다른 상태로 기록한다.

### 6. 제작과 검증은 작업 그래프를 따른다

선행 작업의 실제 출력에서 심볼을 갱신하고 다음 작업의 입력을 만든다. 심볼 연결과 코드 변경은 기존 무결성·쓰기 범위 검사 및 체크포인트를 통과해야 한다.

실패 시 성공한 작업과 계획을 보존한다. 독립된 준비 작업은 계속 수행할 수 있고, 실패 노드의 의존 작업은 blocked로 표시한다. 같은 입력의 무한 재생성은 하지 않는다. 새로운 소스나 명세 오류 등 구체적인 근거가 있을 때 해당 작업의 계획만 수정한다.

완료는 파일 존재나 JAR 컴파일만으로 판정하지 않는다. 요구별 실제 플레이 동작, 이벤트 연결, 상태 변경, 필요한 리소스를 확인한다. 최종 완료는 요구별 검증 결과가 모두 연결된 경우에만 표시한다.

## 기존 코드 재사용과 변경 경계

| 기존 코드 | 재사용할 부분 | 변경할 연결 |
|---|---|---|
| request_scope.py / request_requirements.py | 원본 요구 ID·원문 보존·복합 요구 판별 | 복합/복구 항목을 행동 작업으로 구체화 |
| reuse_discovery.py / grounded_source_reuse.py | 제공자 검색·후보 순위·고정/검증 | 오류 유형 기록, 행동과 심볼 단위 재사용/이식 계획 |
| source_transplant.py | 고정 소스·파일 해시·의존 관계 | 이식 가능한 작은 소스 단위와 원인별 검사 자료 |
| authored_source_rag.py | 작업 소유자별 소스 제한 | AST 경계·정확한 API·호출 관계 전달 |
| api_symbol_extractor.py / jar_api_extractor.py / java_symbol_resolver.py | 실제 JAR 선언·JVM descriptor·javac 호출 해석 | 작업별 대상 타입·오버로드·side·namespace 연결 |
| authored_feature_implementation.py | 요구 ID와 상태·콘텐츠 소유자 연결 | 요구당 대형 콜백 하나 대신 제작 작업에서 생성 |
| first_pass_java_source.py / first_pass_java_declarations.py | 검증된 선언·패키지·클래스 렌더링 | 자유 Java 본문 생성을 연산 그래프 렌더링으로 전환 |
| canonical_generators.py | 등록된 입력/출력·대상·무결성 검사 | 작업 계약이 있는 Java 경로에 새 작성기 연결 |
| residual_generation_contract.py | 쓰기 경로·기존 해시·보호 코드 검사 | 재사용/이식/glue 작업의 변경 범위를 연결 |
| work_graph.py / artifact_graph_executor.py / artifact_job_checkpoint.py | 의존 작업 실행·체크포인트·영수증 | 작업별 검증 상태와 요구 범위 연결 |
| Typed PlanIR·상태 컴파일러 | 이미 검증된 결정적 생성 | 실제 심볼을 신규 작업에 전달, 효과 중복 방지 |

## 실행계획

아래 신규 경로와 인터페이스는 제안하는 변경 사항이다. 현재 저장소에 구현되어 있다고 주장하지 않는다. 실제 코드 작성 전 이 계약을 확인한다.

### Task 1 — 행동 요구와 제작 작업 계약

**Files:** 새 construction_task.py, construction_planner.py. 변경 request_scope.py, authored_feature_implementation.py. 검증 tests/test_construction_tasks.py.

**Interfaces:** construction_task.py에 BehaviorRequirement와 ConstructionTask를 정의한다. ConstructionTask는 위 작업 계약 표의 필드를 가지며 to_dict()/from_dict() 왕복 결과를 검증한다. construction_planner.py의 plan_construction_tasks(scope, source_evidence, target_symbols, router) -> tuple[ConstructionTask, ...]가 요구별 작은 작업을 만든다.

- [ ] test_compound_request_keeps_parent_and_observable_children: 로그의 복합 문장에 부모 ID와 자식 행동이 남고 자식들을 합친 요구 범위가 부모 범위를 누락하지 않는지 확인한다.
- [ ] test_source_recovery_is_not_direct_java_job: 복구 원문이 바로 하나의 구현 작업으로 넘어가지 않는지 확인한다.
- [ ] test_large_plan_has_no_global_task_cap: 65개 독립 행동의 ID와 검증 연결이 모두 남는지 확인한다.
- [ ] 먼저 해당 pytest를 실행해 기존 경로의 실패를 확인한다.
- [ ] 작업 계약과 행동 분해 경로를 구현한다. 분해 실패는 원본 요구를 보존한 상태로 명시한다.
- [ ] 같은 테스트를 통과시키고 main에 계약 및 분해 변경을 커밋한다.

### Task 2 — 소스 검색·검증 실패 상태와 재사용 자료

**Files:** 새 source_reuse_evidence.py. 변경 reuse_discovery.py, grounded_source_reuse.py, source_transplant.py. 검증 tests/test_construction_reuse_evidence.py.

**Interfaces:** source_reuse_evidence.py에 SourceReuseEvidence를 정의한다. build_source_reuse_evidence(requirements, reuse_plan, target_symbols) -> tuple[SourceReuseEvidence, ...]. 레코드는 고정 파일·심볼·설치 경로·상태/호출 연결·남은 동작·검증 및 실패 이유를 갖는다.

- [ ] test_empty_search_and_provider_failure_are_distinct: 0건과 HTTP/제공자 오류가 다른 상태·원인으로 남는지 확인한다.
- [ ] test_target_failure_creates_port_obligations: 구버전 소스의 컴파일 실패가 메서드별 이식 작업으로 연결되고 기능 전체 fresh 작업으로 뭉치지 않는지 확인한다.
- [ ] test_partial_source_reuse_preserves_residual_behavior: 한 기능 일부를 재사용해도 나머지 행동과 glue가 남는지 확인한다.
- [ ] 실패 테스트를 실행한 뒤 기존 해시·라이선스·의존 관계 검사를 유지하면서 구현한다.
- [ ] 테스트 및 원본 소스 자료 연결을 확인하고 main에 커밋한다.

### Task 3 — 작업에 필요한 코드와 대상 API만 전달

**Files:** 새 construction_context.py. 변경 authored_source_rag.py, authored_feature_implementation.py. 검증 tests/test_construction_context.py.

**Interfaces:** bind_task_context(task: ConstructionTask, source_evidence, project_symbols, target_symbols) -> dict. source_evidence는 Task 2 결과다. 반환 필드는 task_id, intent, input_symbols, source_units, state_reads, state_writes, trigger, effects, output_contract, prerequisite_hashes로 한정한다. 전체 요구·전체 상태·이전 완성 본문을 추가하지 않는다.

construction_context.py에 build_target_symbols(classpath, *, java_version: int, namespace: str, classpath_sides) -> dict를 정의한다. 기존 jar_api_extractor.inspect_jar와 MinecraftAPIExtractor를 재사용하여 owner#name+descriptor ID, 접근권한, static 여부, side, namespace를 만든다. 필요한 매핑은 대상 namespace를 명시하여 apply_mappings로 적용한다. 프로젝트 선행 산출물의 호출은 JavaSymbolResolver.extract_symbols_from_source로 javac 해석을 확인한다. 추출 결과가 없는 API 이름을 모델의 사전 지식으로 채우지 않는다.

- [ ] test_context_excludes_unrelated_state_and_completed_bodies: 관련 없는 상태와 이전 완성 코드를 기본 입력에서 제외하는지 확인한다.
- [ ] test_source_selection_keeps_complete_method_boundaries: 긴 메서드 중간에서 문자열을 잘라 코드 계약을 만들지 않는지 확인한다.
- [ ] test_overload_side_and_mapping_binding: 오버로드·클라이언트 side·구버전 이름을 대상 API에 정확히 연결하거나 구체적 이유로 거부하는지 확인한다.
- [ ] 실패 테스트를 실행한 뒤 작업별 입력을 구현한다.
- [ ] 테스트 및 입력 해시/심볼 자료를 확인하고 main에 커밋한다.

### Task 4 — 범용 Java 연산 그래프와 호스트 렌더링

**Files:** 새 java_operation_ir.py, construction_java.py. 변경 first_pass_java_declarations.py. 검증 tests/test_construction_java.py.

**Interfaces:** validate_java_operations(task, nodes, symbols) -> None. render_task_java(task, nodes, symbols) -> str. nodes는 node_id, op, input_refs, typed_result, attributes를 가지는 평면 노드 목록이며 모든 참조·타입·호출을 검증한다. attributes의 허용 필드는 op별로 정의한다. 구현 의미는 모델이 작성하고 구문은 렌더러가 작성한다.

- [ ] test_declaration_injection_cannot_enter_expression: 타입/리터럴/심볼 필드의 ‘private static final ...’ 선언 삽입을 Java 작성 전에 거부하는지 확인한다.
- [ ] test_balanced_syntax_and_real_call_types: 중첩 연산·조건·반복·람다를 렌더링해 javac로 확인한다.
- [ ] test_unknown_operation_has_no_raw_java_fallback: 미지원 연산이 자유 Java 요청으로 복귀하지 않는지 확인한다.
- [ ] test_logic_is_not_limited_to_gameplay_catalog: 기본 구문과 실제 심볼 조합으로 새 계산·상태 알고리즘을 표현하는지 확인한다.
- [ ] 실패 테스트를 실행한 뒤 타입 참조와 구문별 렌더러를 구현한다.
- [ ] Python 검증과 실제 javac 검증 결과를 각각 기록하고 main에 커밋한다.

### Task 5 — 모델 작성 경로와 전체 제작 경로 연결

**Files:** 변경 canonical_generators.py, authored_feature_implementation.py, first_pass_java_source.py. 검증 tests/test_construction_generation_route.py.

**Interfaces:** author_task_operations(router, task, context) -> tuple[dict, ...]를 construction_java.py에 구현한다. 한 번의 요청은 노드 하나 또는 작은 묶음이며, 필요한 참조를 저장된 작업 상태에서 공급한다. 최종 출력은 render_task_java 결과와 기존 canonical 대상·무결성 검사를 통과한다.

- [ ] test_native_coder_receives_only_current_task: 현재 행동과 필요한 심볼만 모델에 전달하는지 확인한다.
- [ ] test_many_tasks_do_not_grow_request_with_all_prior_code: 작업 수가 늘어도 전체 과거 본문을 누적하지 않는지 확인한다.
- [ ] test_legacy_plan_rebuilds_work_contract_before_generation: 구형 계획도 먼저 작업 계약을 만들고 미검증 자유 Java 경로로 빠지지 않는지 확인한다.
- [ ] test_canonical_and_deterministic_guards_remain: 기존 대상/입출력/결정적 백엔드 검사를 우회하지 않는지 확인한다.
- [ ] 실패 테스트를 실행한 뒤 새 작성기를 모든 해당 Java 제작 경로에 연결한다.
- [ ] 회귀 검증 후 main에 커밋한다.

### Task 6 — 재사용 설치·glue·남은 구현의 실제 연결

**Files:** 변경 authored_reuse_bridge.py, residual_generation_contract.py, authored_feature_implementation.py. 검증 tests/test_construction_linking.py.

**Interfaces:** construction_planner의 작업을 통해 설치된 코드의 output_symbols를 이후 input_symbols로 연결한다. bind_task_context는 설치/이식 완료 영수증의 해시를 확인한다. 등록과 거래·보상 효과의 소유자는 하나로 유지한다.

- [ ] test_reuse_is_called_by_generated_feature: 설치 파일 존재 외에 실제 호출·등록·상태 연결을 검사한다.
- [ ] test_no_call_to_reference_only_uninstalled_type: 참고용 구버전 클래스가 그대로 import되지 않는지 확인한다.
- [ ] test_typed_effect_and_glue_do_not_double_charge: 기존 Typed/UI 구매 효과와 새 콜백이 중복 실행되지 않는지 확인한다.
- [ ] 실패 테스트를 실행하고 실제 이식·glue 연결을 구현한다.
- [ ] 대상 프로젝트 컴파일과 해당 동작 검증을 기록한 뒤 main에 커밋한다.

### Task 7 — 실행·재개·완료 판정

**Files:** 변경 work_graph.py, artifact_graph_executor.py, artifact_job_checkpoint.py, complete_orchestrator.py. 검증 tests/test_construction_execution.py.

**Interfaces:** 기존 작업 영수증에 task_id, requirement_ids, output_symbols, verification_ids, 검증 상태를 연결한다. 완료 집계는 성공한 요청 수가 아니라 요구별 검증 자료를 사용한다.

- [ ] test_failure_keeps_successes_and_independent_ready_work: 한 노드 실패가 성공 산출물과 전체 계획을 지우지 않는지 확인한다.
- [ ] test_resume_does_not_duplicate_registration_or_purchase: 재개가 중복 등록/실행을 유발하지 않는지 확인한다.
- [ ] test_build_only_cannot_mark_behavior_complete: 컴파일만 성공한 기능을 동작 완료로 표시하지 않는지 확인한다.
- [ ] 실패 테스트를 실행한 뒤 기존 실행기와 체크포인트를 확장한다.
- [ ] 작업 그래프·원본 요구·실제 영수증의 대응을 확인하고 main에 커밋한다.

### Task 8 — 실제 MiMo 9B 첫 생성부터 원본 요청까지 확인

**Files:** 새 tools/verify_small_model_construction.py. 변경 기존 Colab 실행 경로와 실패 산출물 수집. 검증 자료는 모델 설정·요청·응답·작업/소스 해시·빌드/동작 결과를 보존한다.

**Interfaces:** 실행 도구는 기존 모델 라우터·실제 대상 설정·Task 1~7 경로를 사용한다. 실제 실행 명령의 server-url, 모델 ID, target 설정은 현장 실행에 사용하는 값으로 기록한다. 형식 테스트용 가짜 라우터 결과를 모델 성공으로 집계하지 않는다.

- [ ] 같은 MiMo 9B 설정으로 작은 필드/계산 작업의 생성 → 렌더링 → javac 성공을 확인한다.
- [ ] 실제 대상 Fabric 이벤트의 등록 → 로드 → 동작을 확인한다.
- [ ] 고정 donor 소스의 재사용 또는 메서드 이식 → glue 연결 → 실제 호출을 확인한다.
- [ ] 다른 행동들과 작업이 많은 요청으로 누락·입력 누적·재개 회귀를 확인한다.
- [ ] 첨부 실패와 같은 복합 요구를 새 작업 그래프로 제작하고 요구별 실제 동작을 확인한다.
- [ ] 120초 무진행이나 문법 오류가 재현되면 정확한 원문·문법/스키마 ID·서버 상태를 수집한다. 시간만 늘리거나 동일 입력을 반복하지 않는다.
- [ ] 실제 작은 모델·대상 빌드·동작 결과와 CI 결과를 구분해 기록하고 main에 검증 자료 연결을 커밋한다.

## 완료 판정과 현재 상태

설계 검토에서는 원본 요구 보존, 재사용 자료, 실제 호출 관계, 새 구현 단위, 생성 문법, 모델 정지, 실행/재개, 동작 검증의 담당이 모두 연결되는지 확인한다.

제품 수정 완료는 위 구현과 실제 작은 모델 제작 검증이 끝난 상태다. 현재는 로그 분석과 코드 조사, 복합 요구 판별 및 연결 부재 확인까지 수행했다. 제품 코드 변경, 실제 MiMo 9B 추론, 대상 Minecraft 모드 빌드는 수행하지 않았다.

구현 전 사용자 확인 사항은 이 구조가 요청한 방향과 일치하는지다. 작업 수를 줄이거나 모든 동작을 고정 호스트 기능 목록으로 바꾸는 설계가 아니다.

