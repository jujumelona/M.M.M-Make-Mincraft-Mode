Java 생성 데이터와 실행 도구의 전송 경로
=========================

Java 클래스 계획, import 항목, 선언 메타데이터는 ``ModelRouter.generate_structured_data``로 JSON을 받는다. 구현은 ``generate_java_source``로 완전한 Java 선언 하나를 직접 받는다. 두 경로 모두 실행 도구를 요청하지 않는다. 메타데이터에는 JSON Schema를, 구현에는 Java 구문의 GBNF를 서버 샘플러에 적용한다. Java 소스를 JSON 문자열 안의 자유 문자로 다루지 않는다.

출력 예산과 작업 분할
--------------------

기본 MiMo 프로필은 ``dynamic_output_budget: true``를 사용한다. 이 프로필의 ``max_new_tokens``는 입력 문맥 배치를 위한 예약값이다. 디코딩을 8,192토큰에서 중단하는 상한이 아니다.

출력 정책은 ``llama_server_hardware_policy._server_payload``에서 직접 적용한다. 선택적 설치 함수 실행 여부나 ModelRouter 진입점에 따라 다른 한도를 사용하지 않는다. 호환 설치 함수도 같은 정책에 위임한다. 마지막 전송 단계에서는 서버가 계산한 실제 입력 토큰 수와 슬롯 문맥 크기의 차이를 ``max_tokens``로 사용한다. 기본 프로필에서 내부 요청에 들어 있는 단계별 상한을 다시 적용하지 않는다. 사용자가 명시한 정적 프로필 또는 환경변수 상한은 그대로 존중한다.

Java 생성 데이터에는 출력 최소 예약량을 지정하지 않는다. 스키마가 허용하는 최대 직렬화 크기는 실제 응답에 반드시 필요한 최소 크기가 아니기 때문이다. 이 둘을 동일하게 취급해 ``required=14949 available=8192``로 막던 데이터 생성 경로를 제거했다.

계획과 중첩 타입의 객체·배열은 토큰 숫자에 따라 뒤늦게 분리하지 않는다. 호스트가 처음부터 필드와 배열 항목 단위로 요청하고 조립한다. 작은 선언 메타데이터는 한 요청으로 받고, 구현은 완전한 필드·메서드·생성자 선언 하나로 받는다. 실패한 출력의 재시도나 잘린 코드 이어 붙이기를 작업 분할로 사용하지 않는다.

선언과 구현의 소유권
------------------

먼저 필드·메서드·생성자·중첩 클래스·record·enum의 종류와 필요한 역할을 계획하고, 타입·매개변수·가시성 등 선언 데이터를 확정한다. 이 선언 구문을 Java 파서로 읽은 다음에 구현을 요청하므로, 뒤에 선언되는 메서드도 실제 매개변수와 반환 타입을 보고 호출한다. 응답은 완전한 Java 선언 하나다. GBNF가 고정 서명과 끝 구분자를 소유하며, 안쪽 구현도 해당 Java 시작 규칙을 따라 생성한다. 응답을 다시 ``=`` 뒤나 다른 메서드 본문 안에 넣지 않는다. 패키지·클래스·메서드 서명과 바깥 괄호의 소유권은 호스트에 있다.

공통 request-feature의 ``request_feature_entrypoint`` 역할은 ``net.fabricmc.api.ModInitializer``와 ``public void onInitialize()``에 바인딩한다. 기존 CLIENT canonical 역할은 ``net.fabricmc.api.ClientModInitializer``와 ``public void onInitializeClient()``에 바인딩한다. 이러한 초기화 클래스에서 모델이 상속, 생성자 또는 초기화 메서드 이름을 다시 선택하지 않는다. 내부 helper 이름도 호스트가 선택하고, 필수 초기화 본문은 helper 구현 이후 별도로 생성한다. 이 경로는 저장된 기존 역할 필드로 결정한다.

각 구현 요청에는 기능·API 구현 요구사항, COMMON/CLIENT 구분, 대상 파일, 실제 선행 소스와 선언된 helper API를 전달한다. 생산자가 ``java_implementation_requirements``로 행동 요구사항을 분리한다. 완성 파일의 ``Return complete compilable Java source``와 클래스 선언 출력 지시는 필드·메서드 요청으로 전달하지 않는다. 초기화 역할은 별도의 ``host_entrypoint`` 데이터로 전달한다.

``first_pass_java_declarations``는 기존 Tree-sitter Java 파서로 선택한 종류·이름의 선언 하나를 구성한다. 구현 문자열이 다른 선언이나 클래스 범위로 빠져나오면 그 문자열을 소스에 붙이지 않는다. 반환된 전체 클래스를 정규식으로 잘라 쓰거나 메서드 이름을 바꿔 복구하지 않는다. 소스 슬롯의 기존 계약과 실제 빌드 경로는 유지한다.

서명 뒤에 자유 ``char{...}`` 문자열을 붙이던 ``structured_string_frame``은 제거했다. 필드는 ``_variable_initializer``, 메서드는 ``block`` 내부, 생성자는 ``constructor_body`` 내부를 시작 규칙으로 사용한다. 서버에는 ``grammar``만 적용하고 JSON 형식 문법과 혼합하지 않는다. 구현 응답의 2,600자 제한도 사용하지 않는다. 완성 파일의 기존 슬롯 계약과 실제 문맥 한도는 유지한다.

Java 문법 리소스
----------------

``tools/grammars/tree-sitter-java-0.23.5.json``은 MIT 라이선스의 ``tree-sitter/tree-sitter-java`` v0.23.5 원문이다. admission parser도 같은 버전으로 고정한다. ``tools/compile_java_source_grammar.py``는 nullable·unit production을 제거하고 좌재귀가 있는 SCC에 selective left-corner 변환을 적용해 재사용 가능한 native GBNF를 만든다. 리소스에는 원본 commit·blob SHA와 라이선스를 포함한다. 이는 모델 응답의 복구나 재생성이 아니다.

구문 토큰 사이에는 공백을 넣어 키워드·식별자의 잘못된 결합을 방지한다. 문자열·문자·text block은 하나의 lexeme으로 생성해 공백 삽입으로 문자열 값을 바꾸지 않는다. 식별자는 기존 선언 계약과 같은 ASCII 문자 집합을 사용하고, 문자열 값은 Unicode 원문으로 출력한다. Java의 토큰화 전에 따옴표·개행을 만들 수 있는 Unicode escape는 생성하지 않으며, 표준 escape와 8진 escape를 사용한다. package/import/module 선언은 본문 범위에서 제외한다. 현재 비-preview Java 대상에 없는 string template도 생성하지 않는다. 이 문법은 Java 파서의 구조적 계약을 생성기에 전달하며, 타입·Minecraft/Fabric API·게임 동작의 정확성까지 보장하지 않는다.

원본 파서의 느슨한 숫자 어휘도 생성 시에는 Java 규칙으로 좁힌다. 8진수는 ``0``으로 시작하며 ``0o``·``0O``는 허용하지 않는다. 16진 부동소수점에는 ``p``·``P`` 지수를 필수로 둔다.

생성 제약
---------

- 메타데이터에는 JSON Schema → GBNF, 구현에는 Java GBNF 제약을 생성 단계에 적용한다.
- 서버에 적용하는 메타데이터 스키마와 Java 응답 프레임을 모델의 프롬프트에도 명시한다.
- 호스트가 클래스 정체성, 바깥 괄호, 조립을 담당한다.
- 데이터 응답의 스키마를 해제하는 재생성과 설명문 안의 JSON 추출을 사용하지 않는다.

참조:

1. `llama.cpp GBNF 및 JSON Schema 공식 문서 <https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md>`_
2. `이 프로젝트가 고정한 llama.cpp의 json_schema 처리 구현 <https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/tools/server/server-schema.cpp>`_
3. `Outlines 공식 구현과 Phi-3-mini 적용 예제 <https://github.com/dottxt-ai/outlines>`_
4. `Generating Structured Outputs from Language Models: Benchmark and Studies <https://arxiv.org/abs/2501.10868>`_

이 방식은 선언 소유권과 출력 구조를 다룬다. Java API 선택과 기능 구현의 정확성까지 스키마로 보장되는 것은 아니다. Fabric 초기화 API의 원문은 `ModInitializer <https://github.com/FabricMC/fabric-loader/blob/master/src/main/java/net/fabricmc/api/ModInitializer.java>`_ 와 `ClientModInitializer <https://github.com/FabricMC/fabric-loader/blob/master/src/main/java/net/fabricmc/api/ClientModInitializer.java>`_ 를 따른다.

모델 원문 진단 로그
------------------

실제 요청 본문, 스키마, 출력 예산, 응답 스트림 원문, 종료 사유, 서버의 usage/timings는 요청별 JSONL 파일에 기록한다. 별도의 디버그 옵션을 켤 필요가 없다. 같은 ``request_id``로 기록을 연결한다. ``context``에는 생성 중인 ``job_id``, canonical leaf, 모델 역할과 모델 ID, 데이터 페이지 ``action``이 들어간다. 화면의 ``MODEL COMPLETION TRACE:``에는 요청 시작과 종료 상태, 파일 경로만 짧게 표시한다. 원문과 토큰별 응답 조각을 화면에 출력하지 않는다.

``response_delta``는 파서와 스트림 감시기에 전달하기 전에 파일에 기록한다. 출력에 포함된 공백, 추론 채널, 부분 도구 인자와 잘못된 JSON을 생략하거나 길이로 자르지 않는다. ``response_finished``에는 그때까지 조립한 응답 전체를 남기며, 정상 종료와 미완료, 오류를 구분한다. Artifact graph의 바깥 시간 제한이 만료되면 해당 작업의 활성 요청을 찾아 ``response_snapshot``에 부분 응답 전체와 ``ARTIFACT_GRAPH_WORK_TIMEOUT`` 사유를 파일에 기록한다.

같은 레코드는 요청별 JSONL에도 즉시 기록하고 flush한다. 기본 위치는 ``.mmm/traces/model_completions/<request_id>.jsonl``이며, 기존 root-cause trace 경로를 지정한 실행에서는 해당 저널과 같은 디렉터리의 ``model_completions``에 저장한다. 실제 경로는 각 로그의 ``journal_path``에 표시한다. 파일 기록이 실패하면 ``recording_error``를 로그에 표시한다. HTTP 인증 헤더는 기록하지 않는다.

Colab의 계획 또는 모델 제작 단계가 실패하면 root-cause 저널, 요청별 원문, 진단 JSON과 오류 traceback을 ``OUTPUT_ROOT/diagnostics/planning-logs.zip`` 또는 ``production-logs.zip``으로 압축하고 자동 다운로드한다. 다운로드가 안 되더라도 원문 폴더와 ZIP 경로를 화면에 남긴다. 결과 다운로드 셀에서도 실패 로그 ZIP을 다시 받을 수 있다.

Java 선언 파싱에 실패하면 ``java_declaration_failure`` 이벤트에 종류·이름·metadata/implementation 단계, 최대 1,024자의 실패 멤버 미리보기와 소스 파일 경로를 남긴다. 파서에 실제로 전달한 전체 소스는 ``.mmm/traces/artifacts/*.json``에 저장되고 기존 실패 ZIP에 포함된다. 모델의 토큰별 출력과 전체 원문은 콘솔에 복사하지 않는다.

실행 범위
---------

사용자 요청에 따라 이번 변경의 관련 테스트는 실행하지 않았다. 실제 Colab/T4의 MiMo 추론과 해당 모드의 빌드 성공은 이 환경에서 확인하지 못했다.
