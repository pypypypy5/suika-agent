# 서버 재시작 및 로그 관리 가이드

## 개요

학습 중 Node.js 게임 서버가 크래시하는 문제를 해결하기 위해 다음 기능들이 추가되었습니다:

1. **서버 로그 자동 저장** - 모든 서버 stdout/stderr를 파일로 기록
2. **자동 서버 재시작** - 연결 실패 시 서버 헬스체크 후 재시작
3. **강제 서버 재시작** - 기존 서버를 죽이고 새로 시작 (로그 캡처 보장)

## 1. 서버 로그 자동 저장

### 위치

모든 서버 로그는 `logs/` 디렉토리에 저장됩니다:

```
logs/
├── server_port8924_stdout.log
├── server_port8924_stderr.log
├── server_port8925_stdout.log
├── server_port8925_stderr.log
...
```

### 로그 형식

```
============================================================
[2026-01-27 02:45:08] Starting server on port 8924
============================================================

> suika-game-server@1.0.0 start
> node server.js

Suika Game Server running on port 8924
API endpoints:
  POST /api/reset  - Reset environment
  POST /api/step   - Execute action
  ...
```

### 사용 방법

학습 중 연결 에러가 발생하면:

```bash
# 에러 메시지에서 포트 확인
# Request failed: HTTPConnectionPool(host='localhost', port=8925)

# 해당 포트의 로그 확인
cat logs/server_port8925_stderr.log  # 에러 로그
cat logs/server_port8925_stdout.log  # 일반 로그
```

## 2. 자동 서버 재시작

### 동작 방식

`reset()` 또는 `step()` 실행 시:

1. 요청 실패 감지
2. 서버 health check (`GET /health`)
3. 서버가 죽었으면 재시작 시도
4. 최대 2회 재시작 시도
5. 실패 시 에러 반환 또는 더미 observation 반환

### 코드

```python
# envs/suika_wrapper.py의 reset()과 step()에 자동 구현됨
# 추가 설정 불필요
```

### 로그 메시지

```
Warning: env.step failed (attempt 1/3): HTTPConnectionPool...
Server health check failed. Attempting restart...
[Port 8925] Restarting server...
[Port 8925] Server restarted successfully
```

## 3. 강제 서버 재시작 옵션

### 문제 상황

기존 서버가 이미 실행 중일 때:
- Wrapper가 서버를 시작하지 않음
- 로그를 캡처할 수 없음
- 서버 크래시 시 로그 확인 불가

### 해결책

`force_server_restart=True` 옵션 사용:

```python
env = SuikaEnvWrapper(
    port=8924,
    auto_start_server=True,
    force_server_restart=True  # 기존 서버 강제 종료 후 재시작
)
```

### 동작

1. 포트 사용 중 감지
2. 해당 포트의 프로세스 강제 종료 (psutil 사용)
3. 새 서버 시작
4. 로그 파일에 기록 시작

### 사용 시나리오

#### 개발/디버깅
```python
# 매번 깨끗한 서버로 시작하고 싶을 때
env = SuikaEnvWrapper(
    port=8924,
    force_server_restart=True
)
```

#### 장시간 학습
```python
# config/default.yaml에 추가
env:
  auto_start_server: true
  force_server_restart: true  # 모든 환경에 적용
  auto_start_server_per_env: true
```

## 4. 전체 워크플로우

### 정상 학습 시

```
1. 환경 생성
   ↓
2. 서버 자동 시작 (없을 경우)
   ↓
3. 로그 파일 생성 및 기록 시작
   ↓
4. 학습 진행
   ↓
5. 환경 종료 시 서버도 종료
```

### 서버 크래시 시

```
1. Step/Reset 실패
   ↓
2. Health check 실행
   ↓
3. 서버 죽음 감지
   ↓
4. 서버 재시작
   ↓
5. 로그에 재시작 기록
   ↓
6. 학습 계속 진행
```

## 5. 설정 예시

### 기본 설정 (자동 재시작만)
```yaml
env:
  auto_start_server: true
  auto_start_server_per_env: true
  port_base: 8924
```

### 강제 재시작 포함 (로그 보장)
```yaml
env:
  auto_start_server: true
  auto_start_server_per_env: true
  force_server_restart: true  # 추가
  port_base: 8924
```

### Python 코드에서 직접 설정
```python
from envs import SuikaEnvWrapper

# 기본 사용
env = SuikaEnvWrapper(port=8924)

# 강제 재시작 사용
env = SuikaEnvWrapper(
    port=8924,
    force_server_restart=True
)
```

## 6. 트러블슈팅

### 로그가 생성되지 않는 경우

**원인**: 서버가 wrapper 외부에서 시작되었음

**해결책**: `force_server_restart=True` 사용

```python
env = SuikaEnvWrapper(
    port=8924,
    force_server_restart=True
)
```

### 재시작이 계속 실패하는 경우

**확인사항**:
1. 로그 파일 확인 (`logs/server_port{PORT}_stderr.log`)
2. Node.js 및 npm 설치 확인
3. 포트 충돌 확인
4. 디스크 공간 확인

**해결**:
```bash
# 수동으로 모든 서버 종료
taskkill /F /IM node.exe  # Windows
pkill -f node             # Linux/Mac

# 포트 확인
netstat -ano | findstr :8924  # Windows
lsof -i :8924                 # Linux/Mac
```

### 로그 파일이 너무 커지는 경우

로그 파일은 append 모드로 누적됩니다. 주기적으로 정리:

```bash
# logs 디렉토리 전체 삭제
rm -rf logs/

# 또는 특정 포트만
rm logs/server_port8924_*.log
```

## 7. 구현 세부사항

### 추가된 메서드

**`envs/suika_wrapper.py`**:
- `_is_server_healthy(port)` - 서버 health check
- `_kill_server()` - 현재 서버 프로세스 종료
- `_kill_existing_server_on_port(port)` - 특정 포트의 프로세스 종료
- `_restart_server(port)` - 서버 재시작

### 수정된 메서드

- `__init__()` - `force_server_restart` 파라미터 추가
- `_start_server_if_needed()` - 로그 파일 생성 및 강제 재시작 로직
- `reset()` - 자동 재시작 로직 추가
- `step()` - 자동 재시작 로직 추가
- `close()` - 로그 파일 핸들 정리

### 의존성

- `psutil>=5.9.0` - 프로세스 관리 (requirements.txt에 추가됨)

## 8. 테스트

### 로그 저장 테스트
```bash
python tests/test_server_logging.py
```

### 서버 재시작 테스트
```bash
python tests/test_server_restart.py
```

### 강제 재시작 테스트
```bash
python tests/test_force_restart.py
```

## 결론

이 업데이트로 인해:
- ✅ 서버 크래시 원인을 로그로 파악 가능
- ✅ 학습 중 서버 문제 발생 시 자동 복구
- ✅ 장시간 학습의 안정성 향상
- ✅ 디버깅 시간 대폭 단축
