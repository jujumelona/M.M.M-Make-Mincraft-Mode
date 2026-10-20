# Fabric 이벤트 바인딩의 호스트 책임

`production-logs (18)`의 두 생성 실패는 `JAVA_HOST_BEHAVIOR_EVENT_UNBOUND`였다.
기존 수집기는 `event` 패키지의 인터페이스 자체에 있는 `EVENT` 필드만 인정하여
26.2의 실제 Fabric API에서도 이벤트를 7개만 반환했다. 서버 틱, 블록 파괴,
엔티티 이벤트처럼 별도 소유 클래스가 가진 필드는 빠졌다. 게임 조건을 Fabric
이벤트 이름과 직접 일치시키라는 선택 지시도 함께 잘못되어 있었다.

호스트는 실제 클래스파일의 `Event<Listener>` 필드를 리스너의 SAM과 연결한다.
필드 소유자, 필드 이름, 리스너 인터페이스를 각각 보존하며, 공용 코드에서는
클라이언트 타입을 제외한다. 매개변수, 제네릭 타입 인자, 반환형과 throws는
검증한 서명에서 만들고 등록은 `Owner.FIELD.register(this::handler)`로 작성한다.

선택은 실제 이벤트 소유 클래스와 그 클래스의 필드 순서로 나눈다. 모든 SAM
서명을 매번 보내는 대신 필요한 클래스의 계약을 전달한다. 글자 수 제한으로
API를 잘라내거나 모르는 이벤트를 임의의 이벤트로 바꾸지 않는다.

게임 조건과 실행 훅은 서로 다른 정보다. `void` 콜백은 호스트가 다음 구조로
만들고 작은 AI에는 조건과 효과 본문을 각각 요청한다.

```java
private void handler(ActualInputs inputs) {
    if (this.matchesBehavior(inputs)) {
        this.applyBehavior(inputs);
    }
}
```

조건은 원래 트리거와 상태를 검사해야 하며, 임의로 항상 참으로 바꾸면 안 된다.
반환값이 있는 콜백은 해당 콜백의 결과 의미를 유지해야 하므로 원래 조건 검사와
반환 처리를 콜백 본문에 유지한다. 실제 입력으로도 구현할 수 없는 바인딩은
실패 응답과 원래 트리거를 남기고 실패한다.

## 검증

조건이 거짓일 때 효과가 실행되지 않고, 참일 때 연결되는지 JDK로 컴파일하고
실행하는 회귀 검사가 있다. 실제 Fabric 클래스파일로 이벤트 수집도 검증한다.

추가로 호스트가 발견한 공용 이벤트 각각에 대해 실제 생성 경로로 Java 클래스를
만들어 공식 Minecraft 26.2 클라이언트 JAR, 게시된 라이브러리, Fabric API
0.155.2+26.2 및 Fabric Loader로 컴파일한다. 가짜 Minecraft 클래스는 사용하지
않는다. 다운로드는 공식 메타데이터/체크섬으로 검증한다.

```bash
python tools/verify_fabric_event_catalog.py --cache-dir /tmp/mmm-fabric-event-proof
```

JDK 25 이상이 필요하다. 이번 검증은 공용 이벤트 111개의 실제 생성 클래스와
등록 코드를 모두 컴파일했다. 본문은 컴파일 검사 전용 중립 본문이므로 게임
기능의 성공률이나 실제 MiMo 응답, Minecraft 내 동작을 인증하지 않는다.
그 범위는 생성 후 기존 컴파일·GameTest·실행 검증 단계에서 확인해야 한다.
