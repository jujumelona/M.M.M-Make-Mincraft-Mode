Java 생성 데이터와 실행 도구의 전송 경로
=========================

Java 클래스 계획, import 항목, 클래스 멤버는 실행할 도구가 아니라 호스트가 조립할 데이터다. 이 값들은 ``ModelRouter.generate_structured_data``를 통해 ``tools=()``와 ``response_schema``를 사용한다. 생성 서버가 스키마를 토큰 샘플링에 적용하고, 호스트가 반환한 정확한 JSON을 읽는다.

출력 예산과 작업 분할
--------------------

기본 MiMo 프로필은 ``dynamic_output_budget: true``를 사용한다. 이 프로필의 ``max_new_tokens``는 입력 문맥 배치를 위한 예약값이다. 디코딩을 8,192토큰에서 중단하는 상한이 아니다.

출력 정책은 ``llama_server_hardware_policy._server_payload``에서 직접 적용한다. 선택적 설치 함수 실행 여부나 ModelRouter 진입점에 따라 다른 한도를 사용하지 않는다. 호환 설치 함수도 같은 정책에 위임한다. 마지막 전송 단계에서는 서버가 계산한 실제 입력 토큰 수와 슬롯 문맥 크기의 차이를 ``max_tokens``로 사용한다. 기본 프로필에서 내부 요청에 들어 있는 단계별 상한을 다시 적용하지 않는다. 사용자가 명시한 정적 프로필 또는 환경변수 상한은 그대로 존중한다.

Java 생성 데이터에는 출력 최소 예약량을 지정하지 않는다. 스키마가 허용하는 최대 직렬화 크기는 실제 응답에 반드시 필요한 최소 크기가 아니기 때문이다. 이 둘을 동일하게 취급해 ``required=14949 available=8192``로 막던 데이터 생성 경로를 제거했다.

계획의 객체와 배열은 토큰 숫자에 따라 뒤늦게 분리하지 않는다. 호스트가 처음부터 필드와 배열 항목 단위로 요청하고 조립한다. Java 멤버는 완결된 메서드/필드 단위로 생성한다. 실패한 출력의 재시도나 잘린 코드 이어 붙이기를 작업 분할로 사용하지 않는다.

생성 제약
---------

- llama.cpp의 JSON Schema → GBNF 제약을 생성 단계에 적용한다.
- 서버에 적용하는 스키마를 모델의 프롬프트에도 명시한다.
- 호스트가 클래스 정체성, 바깥 괄호, 조립을 담당한다.
- 데이터 응답의 스키마를 해제하는 재생성과 설명문 안의 JSON 추출을 사용하지 않는다.

참조:

1. `llama.cpp GBNF 및 JSON Schema 공식 문서 <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md>`_
2. `이 프로젝트가 고정한 llama.cpp의 json_schema 처리 구현 <https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-schema.cpp>`_
3. `Outlines 공식 구현과 Phi-3-mini 적용 예제 <https://github.com/dottxt-ai/outlines>`_
4. `Generating Structured Outputs from Language Models: Benchmark and Studies <https://arxiv.org/abs/2501.10868>`_

이 방식은 출력 형식의 안정성을 다룬다. Java API 선택과 기능 구현의 정확성까지 스키마로 보장되는 것은 아니다.

실행 범위
---------

사용자 요청에 따라 이번 변경의 관련 테스트는 실행하지 않았다. 실제 Colab/T4의 MiMo 추론과 해당 모드의 빌드 성공은 이 환경에서 확인하지 못했다.
