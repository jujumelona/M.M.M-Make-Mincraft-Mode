# MMM — CurseForge-only 소스 재사용·작은 AI 판단 계약 V1

상태: **구현·검증 기준 문서**  
작성일: 2026-10-18  
적용 범위: M.M.M-Make-Mincraft-Mode / 사용자 선택 Minecraft 버전·로더  
배포 정책: 공식 CurseForge 출처로부터 확인한 코드만 검증 절차를 거쳐 재사용한다. 일반 GitHub 검색은 사용하지 않는다.

## 0. 목적과 기존 문제

사용자의 자연어 요구를 기능(feature/capability) 단위로 분할하고, CurseForge에서 기존 모드를 검색하고, **실제 소스 코드**를 검사한 다음 작은 AI가 요구 기능에 대한 **구체적 근거**로 재사용 또는 수정·신규 구현을 판단하게 한다. 소스가 발견되지 않았거나 라이선스·실행 검증이 불가능하면 검증 성공으로 표현하지 않는다.

기존 커밋 `30d65e7545b9`에서 `reuse_discovery.py`는 후보를 이름/설명/카테고리의 키워드 일치로 정렬하며 `source_url`을 단축 후보에 전달하지 않았다. `grounded_source_reuse.py`는 모든 capability를 `mode=fresh`로 고정하고 `inspected=0`, `proof_attempts=0`을 출력했다. 따라서 키워드 검색 품질을 높이는 것만으로 소스 재사용은 작동하지 않는다.

## 1. 변경 불가능한 원칙

1. **탐색 제공자는 CurseForge 단독**. GitHub 전체 검색·저장소 자동 발굴·빈 검색어·무조건적인 대규모 후보 탐색 금지. 단, CurseForge 공식 `sourceUrl`이 명시한 GitHub/GitLab 저장소를 **직접 열람**하는 것은 소스 추적이지 검색이 아니다.
2. 대상 `minecraft_version`, `loader`는 사용자가 선택한 값. 임의 버전 고정·다른 로더로 은밀히 대체 금지. 구 버전의 *참고 코드*는 이식 후보가 될 수 있으나 사용자가 선택한 빌드에서 검증되어야 한다.
3. 모드의 이름·요약·다운로드 수·키워드 점수는 **카탈로그 관련성 신호**일 뿐 재사용 가능성의 증거가 아니다.
4. 작은 AI의 역할은 **의미 판단**: 실제 소스의 어떤 파일/클래스/메서드가 어떤 요구 기능을 구현하는지, 얼마만큼 재사용·수정 가능한지, 부족한 기능이 무엇인지 판단한다.
5. 호스트의 역할은 **확인 가능한 사실**: 소스 출처, pinned commit, 각 파일의 SHA-256, 라이선스, 대상 버전/로더, 의존성, 컴파일, 기능·런타임 테스트를 검증한다.
6. 작은 AI의 긍정 판단은 `semantic_candidate`일 뿐이다. **실제 재사용 `source_transplant`는 검증된 복사·이식과 대상 빌드 증거가 있기 전까지 금지**. 동작 검증은 별도 단계이며 미실행 시 명확히 미검증으로 기록한다. 자의적 `proof_receipt` 생성 금지.
7. 실패나 근거 없음은 `unverified`/`inspection_error`/`license_blocked`/`source_unavailable`로 추적한다. 실패를 성공으로 승격하거나 무한 검색/반복 계획 생성을 하지 않는다.
8. 라이선스가 불명확한 코드는 복사하지 않는다. 라이선스 정책은 프로젝트의 `reuse_license.py`를 따른다. 저작권 고지와 2차 배포 의무를 보존한다.
9. 후보 검색 → 소스 수집 → 의미 판단 → 재사용 컴파일 → 제작/기능 검증의 각 단계는 구별되는 로그·영수증을 작성한다.

## 2. 판단 기준: 작은 AI (0~5 등급, 증거 필수)

| 기준 | 가중치 | 0점 예 | 5점 예 |
|---|---:|---|---|
| `functional_fit` 기능적 일치성 | 30% | 요청한 동작과 관계없는 코드 | 요청한 핵심 동작을 직접 구현 |
| `partial_reuse` 부분 재사용 | 20% | 활용 가능한 단위 없음 | 독립적으로 추출 가능한 클래스/알고리즘 다수 |
| `completeness` 구현 완성도 | 15% | 선언/스텁뿐 | 주요 실행 경로와 상태 처리 포함 |
| `portability` 버전·API 이식 가능성 | 15% | 현재 타깃에서 이식 실마리 없음 | 유사 API/명확한 변환 경로 |
| `dependency_simplicity` 결합도·의존성 | 10% | 거대한 폐쇄 결합, 다수 누락 의존성 | 독립적 소수 타입·API |
| `net_benefit` 재사용 실익 | 10% | 새로 만드는 것보다 비쌈 | 리스크·작업량을 의미 있게 절감 |

`semantic_score = 20 × (0.30 × functional_fit + 0.20 × partial_reuse + 0.15 × completeness + 0.15 × portability + 0.10 × dependency_simplicity + 0.10 × net_benefit)`.

점수만 높다고 재사용 확정 금지. 작은 AI는 *실제 존재하는* `selected_file`/`selected_symbol`, 구현 증거, 필요한 변경사항과 미구현 잔여 작업을 반드시 제출한다. 상상한 파일/클래스/메서드 또는 모드 소개글만을 근거로 승인 금지. 선별 기능과 코드는 1:1 이상으로 대응시킨다. 자동차 조종 코드를 우주선에 응용하는 것처럼 **이름은 달라도 기능이 유사한 코드**를 정당하게 선택할 수 있다.

권장 분기: `reject`, `reference`, `adapt`, `reuse`의 **의미상 의도**만 선택하고 실제 최종 `mode`는 검증 엔진이 결정한다. 코드 조각 또는 소스 확보가 불가능하면 재사용 판단 대상에서 제외하고 제작으로 넘어간다.

## 3. 고정 파이프라인과 영수증

1. **요구 분해:** 원문 요구를 잃지 않고 고유 기능으로 분리; 기능마다 완수 조건 작성.
2. **CurseForge 후보 검색:** 선택된 버전/로더를 검색 매개변수로 사용. 검색어는 비어 있지 않은 핵심 기능 명사. 캐시/중복 제거, 관련도 점수는 사전 정렬에만 활용.
3. **공식 출처 보존:** `catalog_id`, `source_id`, `catalog_url`, `source_url`, `target_file_indexes`, `matched_queries` 기록. 원문 출처를 임의 생성하지 않는다.
4. **소스 실제 검사:** CurseForge `sourceUrl`이 직접 지정한 안전한 HTTPS 호스트에만 접근. 링크가 Git 호스팅이라면 정확한 저장소를 pinned commit으로 고정하고 다운로드. 아카이브 용량·경로·파일 수 제한; Java 소스 경로·선언과 라이선스 추출. 외부 소스 내 지시는 **모두 신뢰할 수 없는 데이터** 취급.
5. **작은 AI 의미 평가:** 각 기능별로 검사된 소스 조각만 제공. 위 여섯 기준, 실제 파일/심볼, 변경·미구현 목록, 판단 근거를 한 건의 엄격한 JSON 응답으로 수집. 전혀 소스가 없는 모드는 제외.
6. **증거 대조:** 반환한 파일과 심볼이 다운로드한 스냅샷에 존재하는지 호스트가 검사. 없는 항목을 만들어 낸 응답은 거부. AI의 점수가 라이선스·빌드 검증을 대체하지 못한다.
7. **이식·컴파일:** 검증 가능한 소스 조각·의존성을 대상 Fabric 프로젝트의 격리 빌드에 적용. 코드의 원본/수정본 파일 SHA, 컴파일 결과, 오류와 변경 명세를 보존. 구 버전에서 최신 타깃으로 이식 시 마이그레이션 작업으로 기록.
8. **동작 검증:** 실제 구현에 바인딩된 기능 테스트, 가능 시 GameTest/서버 실행·클라이언트 테스트. 테스트를 실행하지 않았으면 `UNVERIFIED`.
9. **결과 반영:** 대상 빌드 검증을 통과한 것만 `source_transplant`, 검증되지 않은 긍정 후보는 `reference_only`, 나머지는 `fresh`. 어떤 경우에도 전체 사용자 요구가 누락되지 않도록 잔여 작업을 코더에 전달.

## 4. 권장 스키마

```json
{
  "capability": "vehicle steering with keyboard input",
  "provider": "curseforge",
  "catalog_id": 12345,
  "source_url": "https://github.com/official-owner/official-project",
  "source_revision": "40-char pinned commit hash",
  "license_spdx": "MIT",
  "source_file": "src/main/java/.../VehicleController.java",
  "source_sha256": "sha256:<hex>",
  "selected_symbol": "VehicleController",
  "semantic_scores": {"functional_fit": 5, "partial_reuse": 4, "completeness": 4, "portability": 3, "dependency_simplicity": 3, "net_benefit": 4},
  "semantic_score": 80,
  "semantic_intent": "adapt",
  "rationale": "what exact implementation does / doesn't cover",
  "residual_work": "what must still be implemented",
  "inspection_status": "inspected",
  "proof_status": "NOT_RUN",
  "final_mode": "fresh"
}
```

위 JSON은 **형식 예시**이며 실제 후보로 오인하면 안 된다. 점수도 예시일 뿐 확정된 임계값이 아니다.

## 5. 검증 게이트 / 통과 조건

- **Search:** 선택한 버전·로더/비어 있지 않은 키워드가 API 요청에 전달됐는가?
- **Source:** 선택된 후보의 실제 `source_url`이 유지되고, 제한된 아카이브에서 Java 소스가 확인되는가?
- **License:** 프로젝트의 허용 SPDX 및 저작권 고지를 보존하는가?
- **AI:** AI가 여섯 항목, 존재하는 파일/심볼, 적합성·마이그레이션·잔여 작업을 제시했는가? 관련 없는 코드도 실제로 탈락하는가?
- **Proof:** 타깃에서 격리 컴파일/해당 기능 테스트가 확인되는가?
- **Final:** `source_transplant` 수 = 대상 파일에 실제 설치됐고 proof 검증된 소스 수. 0개여도 정직하게 실패 이유·잔여 작업을 보고하고 생성 자체는 지속.

## 6. 필수 회귀 테스트

- CurseForge 검색만 호출, 다른 공급자나 빈 검색어 없음.
- 키워드가 일치하지만 Java 소스 없는 후보는 **선택 금지**.
- 공식 sourceUrl이 있는 후보는 선정 점수와 관계없이 **실제 소스 검사 단계로 진입**.
- AI가 존재하지 않는 파일/심볼을 주장하면 거부.
- 허용되지 않은/불명 라이선스는 코드 이식 금지.
- 다른 버전 참고 코드는 `reference` 가능, 사용자 선택 타깃의 컴파일 증거 없이 `source_transplant` 금지.
- 컴파일 실패/타임아웃/테스트 미실행을 PASS로 출력 금지.
- 기록상 `inspected`, `proof_attempts`는 실제 실행된 건수로 계산하고 고정 숫자로 출력 금지.
- 동일 소스/동일 요구 중복 검사 회피; 문맥 바뀐 경우 캐시 무효화.
- 관련 후보 모두 부적합해도 새 구현으로 넘어가며 사용자 요구를 조용히 삭제하지 않음.

## 7. 이번 전달물의 구현 범위

패치 패키지는 현재 저장소에서 **키워드 후보→CurseForge 공식 소스 URL 유지→실제 Java 파일 추출→라이선스 검사→작은 AI 근거 기반 평가**를 연결한다. 설치하지 않은 검증 및 재사용을 성공 처리하지 않는다. 기존 `reuse_proof_executor.py`/`source_transplant.py`의 **검증된 소스 설치 계약은 그대로 유지**한다.

**한계:** 별도 Minecraft/Gradle 빌드 환경 없이 실제 donor 이식·GameTest를 성공시킬 수 없다. 그런 증거가 없는 후보는 `source_transplant`로 승격되지 않으며 실제 코드 재사용 자동 완성은 추가 통합·현장 검증이 필요하다. 사용자의 GitHub 저장소는 이 전달물 작성 중 변경하지 않는다.

## 8. 사람이 다시 보더라도 적용할 규칙

이 문서를 단일 목표 계약으로 취급한다. 다음 작업에서 먼저 이 문서를 읽고, 기능 검색·AI 평가·실제 검증·미구현 작업 넘기기의 전 단계가 통과했는지 확인한다. 모드 검색 건수/제한만 조정하거나 단순히 `mode=fresh`로 치환하여 문제를 덮는 수정을 성공으로 인정하지 않는다. 더 이상 근거 없는 최종 PASS를 출력하지 않는다.


## 구현 현황 (2026-10-18)

- CurseForge 검색 → 공식 소스 URL 검사 → 작은 모델의 의미 평가 → 고정 소스 해시 → 격리 Gradle 컴파일 검사 → 검증된 소스 이식 경로가 연결됨.
- 정해진 파일 수/후보 수 안전 제한 및 검증 실패시 `fresh`로 전환한다.
- 실운영 CurseForge API와 사용자 Colab의 Gradle/GameTest 전체 E2E는 별도 실환경 검증이 필요하며, 단순 테스트 통과를 실운영 성공으로 표현하지 않는다.
