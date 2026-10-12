Java 생성 데이터와 실행 도구의 전송 경로
=========================

Java 클래스 계획, import 항목, 클래스 멤버는 실행할 도구가 아니라 호스트가 조립할 데이터다. 이 값들은 ``ModelRouter.generate_structured_data``를 통해 ``tools=()``와 ``response_schema``를 사용한다. 생성 서버가 스키마를 토큰 샘플링에 적용하고, 호스트가 반환한 정확한 JSON을 읽는다.

수정 전 실패
-------

``cefced5`` 실행 로그에서는 import 데이터 한 항목이 ``tools=1``, ``max_tokens=1740``으로 요청된 뒤 ``finish_reason=length``로 실패했다. JSON 문자열의 최대 크기로 출력 예산을 계산했지만, 실제 요청에는 데이터 응답용 ``json_schema``가 없었다. 데이터 대신 MiMo의 함수 호출 프로토콜을 생성하는 경로였다. 이전 테스트는 이미 완성된 함수 인자를 반환하는 가짜 라우터를 사용해 이 차이를 검증하지 못했다.

새 테스트는 실제 ModelRouter와 LlamaCppAdapter가 구성하는 최종 요청을 확인한다. 도구 선언 없이 응답용 스키마가 전송되어야 하며, 생성 실패 후 스키마 제약을 해제하는 재생성이나 설명문 안의 JSON 추출은 이 경로에서 허용하지 않는다. 일반 실행 도구의 기존 경로는 별도로 유지한다.

적용한 표준 방식
---------

- llama.cpp의 JSON Schema → GBNF 생성 제약을 사용한다. 새 생성 라이브러리는 필요 없다.
- 서버에 적용하는 스키마를 모델의 프롬프트에도 명시한다. 스키마는 자동으로 프롬프트에 삽입되지 않기 때문이다.
- 호스트는 클래스 정체성, 바깥 괄호, 조립을 담당한다. 모델은 필요한 작은 계획 항목과 완결된 클래스 멤버를 생성한다.
- 큰 응답 구조는 추론 전에 분할한다. 페이지 크기 계산은 예외를 제어 흐름으로 사용하지 않는다.
- 스키마에 근거한 출력 예산을 최종 전송 지점까지 유지한다.

참조:

1. `llama.cpp GBNF 및 JSON Schema 공식 문서 <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md>`_
2. `이 프로젝트가 고정한 llama.cpp의 json_schema 처리 구현 <https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-schema.cpp>`_
3. `Outlines 공식 구현과 Phi-3-mini 적용 예제 <https://github.com/dottxt-ai/outlines>`_
4. `Generating Structured Outputs from Language Models: Benchmark and Studies <https://arxiv.org/abs/2501.10868>`_

이 방식은 출력 형식의 안정성을 다룬다. Java API 선택과 기능 구현의 정확성까지 스키마로 보장되는 것은 아니다.

검증 범위
-----

관련 회귀 테스트 110개가 통과했다. 잘못된 도구 전송, 스키마를 해제하는 재생성, 설명문에서 JSON을 추출하는 경로를 각각 실패 테스트로 확인한 뒤 수정했다. 전체 pytest는 이 실행 환경의 starlette 의존성 부재로 수집 단계에서 중단됐다. 실제 Colab/T4의 MiMo 추론과 해당 모드의 빌드 성공은 이 환경에서 확인하지 못했다.
