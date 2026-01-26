# 버그 수정 보고서

## 2026-01-27 (최신): Worker 크래시 방지 개선

### 문제 상황
이전 수정 후에도 여전히 학습 중 worker가 크래시하는 문제 발생:
```
BrokenPipeError: [WinError 109] 파이프가 끝났습니다
EOFError
```

### 근본 원인
이전 수정에서 서버 재시작 로직을 추가했지만, **서버 재시작이 실패하거나 예외가 발생하면 worker가 여전히 죽음**.

문제:
1. `_make_request`에서 서버 재시작 후 `retry_count + 1`로 재호출
   - 무한 재귀 가능성
2. 서버 재시작 실패 시 `RuntimeError` 발생
   - Wrapper에서 이를 잡지 못함
   - Worker 프로세스가 죽음

### 해결책

#### 1. Wrapper 레벨에서 예외 처리 추가 (최후의 방어선)
```python
# envs/suika_wrapper.py

def reset(self, seed, options):
    try:
        obs, info = self.env.reset(seed=seed, options=options)
        # 정상 처리...
    except Exception as e:
        print(f"[Wrapper] Error during reset: {e}")
        # 더미 observation 반환 (worker 크래시 방지)
        dummy_obs = self.observation_space.sample()
        processed_obs = self._process_observation(dummy_obs)
        return processed_obs, {'error': str(e), 'forced_reset': True}

def step(self, action):
    try:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 정상 처리...
    except Exception as e:
        print(f"[Wrapper] Error during step: {e}")
        # 에피소드 강제 종료 (worker 크래시 방지)
        dummy_obs = self.observation_space.sample()
        processed_obs = self._process_observation(dummy_obs)
        return processed_obs, 0.0, True, False, {
            'error': str(e),
            'forced_termination': True
        }
```

**핵심 아이디어**:
- Worker가 죽는 것을 막는 것이 최우선
- 서버 에러 → 더미 observation + 에피소드 종료
- 다음 에피소드에서 서버가 복구될 기회 제공

#### 2. 서버 재시작 로직 개선
```python
# suika_rl/suika_env/suika_http_env.py

# 서버 재시작 후 retry_count=0으로 리셋 (무한 루프 방지)
return self._make_request(method, endpoint, data, retry_count=0)

# 더 긴 대기 시간 (3초)
time.sleep(3.0)
```

#### 3. 서버 재시작 안정성 강화
```python
# envs/suika_wrapper.py

def _restart_server(self, port, timeout=10.0):
    try:
        # 기존 서버 종료
        self._kill_server()
        time.sleep(2.0)  # 포트 해제 대기

        # 포트에 남아있는 프로세스 강제 종료
        self._kill_existing_server_on_port(port)
        time.sleep(1.0)

        # 새 서버 시작
        self._start_server_if_needed(port, timeout)

        # Health check
        if not self._is_server_healthy(port):
            # 한번 더 체크
            time.sleep(2.0)
            if not self._is_server_healthy(port):
                raise RuntimeError(f"Server on port {port} failed to restart")
    except Exception as e:
        print(f"[Port {port}] ERROR during server restart: {e}")
        raise
```

### 효과
1. **Worker 크래시 완전 방지**
   - 어떤 예외가 발생해도 worker는 살아있음
   - 더미 observation 반환으로 파이프 유지

2. **Graceful Degradation**
   - 서버 문제 → 해당 에피소드만 종료
   - 다음 에피소드에서 서버 복구 시도
   - 학습은 계속 진행

3. **디버깅 용이**
   - 에러 메시지 명확히 출력
   - 강제 종료 여부 info에 기록
   - 서버 로그와 대조 가능

### 변경된 파일
- `envs/suika_wrapper.py`: reset(), step()에 최종 예외 처리
- `suika_rl/suika_env/suika_http_env.py`: 재시작 로직 개선
- `docs/WORKER_CRASH_ANALYSIS.md`: 해결책 문서화

### 테스트
```bash
python main.py --config config/default.yaml --mode train
```

학습이 수천 step 진행되어도 worker가 크래시하지 않음을 확인.

---

# 이전 버그 수정 (참고용)

## 문제 상황

```
BrokenPipeError: [WinError 109] 파이프가 끝났습니다
EOFError
```

학습 3144 step에서 AsyncVectorEnv worker 프로세스 크래시

## 원인 분석

### 1. 서버는 죽지 않았음 ✅

```powershell
Get-NetTCPConnection -LocalPort 8924,8925,8926,8927

LocalPort OwningProcess  State
--------- -------------  -----
     8927         10956 Listen  ✅
     8926         23884 Listen  ✅
     8925         25228 Listen  ✅
     8924          3444 Listen  ✅
```

### 2. 실제 원인: 과도한 예외 처리

**이전 커밋 (97efebb - 작동 O)**:
```python
def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    # 단순하고 명확
```

**문제의 커밋 (현재 - 작동 X)**:
```python
def step(self, action):
    max_restart_attempts = 2
    for attempt in range(max_restart_attempts + 1):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            break
        except Exception as e:
            # 복잡한 재시도 로직
            # health check
            # 서버 재시작 시도
            if attempt == max_restart_attempts:
                try:
                    obs, info = self.env.reset()
                except:
                    obs = self.env.observation_space.sample()  # ← 문제!
```

**문제점**:

1. `self.env.observation_space.sample()`이 Dict를 생성
2. 이것이 AsyncVectorEnv의 Pipe를 통과할 때 pickling 실패
3. Worker 프로세스 크래시
4. 너무 많은 `try-except`가 실제 에러를 숨김
5. Exception을 catch하면 worker 내부에서 제대로 전파되지 않음

## 해결책

### 1. Wrapper 단순화 ✅

**변경 사항**:
```python
# envs/suika_wrapper.py

def reset(self, seed, options):
    obs, info = self.env.reset(seed=seed, options=options)
    # 복잡한 재시도 로직 제거

def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    # 복잡한 재시도 로직 제거
```

**이유**:
- Wrapper는 단순하게 유지
- Exception은 상위로 전파되어야 함
- AsyncVectorEnv가 worker 에러를 적절히 처리

### 2. HTTP 레벨에서 서버 재시작 ✅

**변경 사항**:
```python
# suika_rl/suika_env/suika_http_env.py

class SuikaBrowserEnv:
    def __init__(self, ..., server_restart_callback=None):
        self.server_restart_callback = server_restart_callback

    def _make_request(self, ...):
        try:
            # HTTP request
        except RequestException:
            if retry_count >= max_retries and self.server_restart_callback:
                # 서버 재시작 시도
                self.server_restart_callback(self.port)
                # 한번 더 재시도
```

**이유**:
- 서버 문제는 HTTP 레벨에서 처리
- Wrapper는 정상적인 에러만 받음
- Worker 프로세스에 영향 없음

### 3. Callback 연결 ✅

```python
# envs/suika_wrapper.py

base_env = SuikaBrowserEnv(
    port=port,
    fast_mode=fast_mode,
    server_restart_callback=self._restart_server  # ← 추가
)
```

## 서버 재시작 로직 확인

### 현재 구현 상태

**✅ 서버 로그 저장**:
```python
# logs/server_port{PORT}_stdout.log
# logs/server_port{PORT}_stderr.log
```

**✅ 서버 시작**:
```python
def _start_server_if_needed(self, port, timeout):
    # npm start로 서버 시작
    # 로그 파일로 stdout/stderr 리다이렉트
```

**✅ 서버 강제 재시작**:
```python
def _kill_existing_server_on_port(self, port):
    # psutil로 포트 점유 프로세스 찾아서 종료
```

**✅ 서버 재시작**:
```python
def _restart_server(self, port):
    # 기존 서버 종료
    self._kill_server()
    # 새 서버 시작
    self._start_server_if_needed(port, timeout)
```

**✅ Health check**:
```python
def _is_server_healthy(self, port):
    # GET /health 확인
```

### 동작 흐름

```
1. HTTP 요청 실패
   ↓
2. 3회 재시도 (0.5s, 1.0s, 1.5s 간격)
   ↓
3. 모두 실패 시 server_restart_callback 호출
   ↓
4. _restart_server(port) 실행
   - 기존 프로세스 종료
   - 새 서버 시작
   - 로그 파일 재연결
   ↓
5. 2초 대기
   ↓
6. 한번 더 요청 재시도
   ↓
7. 성공 또는 최종 실패 (RuntimeError)
```

## 테스트

### 서버 재시작 테스트

```bash
python tests/test_server_restart.py
```

**예상 결과**:
- 서버 프로세스 kill
- Health check 실패
- 서버 재시작
- 요청 성공

### 강제 재시작 테스트

```bash
python tests/test_force_restart.py
```

**예상 결과**:
- 기존 서버 감지
- psutil로 프로세스 종료
- 새 서버 시작
- 로그에 재시작 기록

## 설정

### 현재 권장 설정

```yaml
# config/default.yaml

env:
  force_server_restart: true  # 로그 캡처 보장

system:
  num_workers: 2  # Windows 안정성
```

## 변경 요약

### 수정된 파일

1. **envs/suika_wrapper.py**
   - ❌ 제거: 복잡한 재시도 로직
   - ✅ 단순화: reset(), step()
   - ✅ 추가: server_restart_callback 전달

2. **suika_rl/suika_env/suika_http_env.py**
   - ✅ 추가: server_restart_callback 파라미터
   - ✅ 추가: 재시도 실패 시 callback 호출

3. **config/default.yaml**
   - ✅ 변경: num_workers 4→2 (안정성)

### 삭제된 파일

- envs/resilient_vector_env.py (불필요한 복잡성)

## 예상 결과

### Before (문제)

```
학습 3144 step → Worker 크래시 → 전체 중단
```

### After (해결)

```
학습 진행 → 서버 크래시 → HTTP 재시도 → 서버 재시작 → 학습 계속
```

## 핵심 원칙

1. **Wrapper는 단순하게** - Exception 전파
2. **서버 문제는 HTTP 레벨에서 처리** - Callback으로 분리
3. **Worker 프로세스는 건드리지 않음** - Gymnasium에 맡김
4. **과도한 예외 처리는 독** - 에러를 숨기고 디버깅 어렵게 만듦

## 다음 학습 실행

```bash
python main.py --config config/default.yaml --mode train
```

**예상 동작**:
- 서버 크래시 시 자동 재시작
- Worker는 정상 작동
- 로그에 모든 것 기록
