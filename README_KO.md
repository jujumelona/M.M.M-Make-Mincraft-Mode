# M.M.M Make Minecraft Mode

[English](README.md)

M.M.M은 Minecraft Java Fabric 1.20.1용 실패 폐쇄형 전체 제작 시스템입니다. 기본 경로는 하나의 승인된 멀티모달 요구를 완전 계획, 신규 생성 또는 기존 소스 수정, 자동 수리, Gradle, GameTest, JAR 검사, 임시 런타임 플레이테스트, 화면 검토, 릴리스 패키징까지 연결합니다.

## 기본 전체 제작 경로

```bash
python -m pip install -e '.[dev,ui,local-model,rag,image,speech,training,production-audio]'
mmm plan "완전한 모드를 만들어줘" --profile t4_local --save complete-proposal.json
mmm validate-proposal complete-proposal.json
mmm execute complete-proposal.json --approve <sha256> --output mmm-output --server-launcher <fabric-server-launch.jar> --accept-eula --screenshot <runtime.png>
```

완전 제안서는 아이템, 블록, 음식, 무기, 도구, 방어구, 작물, 기계, 효과, 인챈트, 명령어, 레시피, 발전과제, 전리품, 애니메이션 엔티티, 퀘스트, 직업, 스킬, 경제, 상점, GUI/네트워크, 파티/길드, 구조물, 오디오와 제한형 커스텀 Java 모듈을 담습니다. 의존 관계와 합격 조건도 승인 해시에 포함됩니다.

## 실제 연결된 과정

- 역할별 Qwen 계획·코딩·RAG, FLUX 자산, Whisper 입력
- 파일 SHA-256 전제 트랜잭션 패치와 유한 자동 수리
- 확장 콘텐츠와 게임 시스템의 실제 Fabric 등록
- 퀘스트·직업·경제·파티 영속 저장
- GeckoLib 엔티티·렌더러·속성·AI goal·애니메이션 연결
- gzip structure NBT·Jigsaw·structure set·월드 리소스
- OGG 생성/가져오기·`sounds.json`·자막·`SoundEvent` 등록
- JDT LS·Gradle·GameTest·독립 JAR 검증
- 임시 Minecraft 1.20.1 서버/클라이언트·Mineflayer·시각 검토
- 소스/JAR 배포 묶음과 검토된 Modrinth/CurseForge 업로드
- 실행 Skill 21개와 실제 FastMCP 서버

## 검증 경계

일반 GitHub CI는 CPU 계약을 검증합니다. 실제 T4 추론, Minecraft, Blockbench와 배포는 검토된 self-hosted `Production integration` workflow가 필요합니다. 실행 파일, EULA 승인, endpoint, credential 또는 증거가 없으면 성공으로 표시하지 않고 실패합니다.

전체 구조는 [docs/PRODUCTION_STACK.md](docs/PRODUCTION_STACK.md)를 참고하십시오.
