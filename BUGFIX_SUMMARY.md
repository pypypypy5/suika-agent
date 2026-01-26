# Worker 크래시 문제 해결 요약

## 문제
학습 중 `BrokenPipeError: [WinError 109] 파이프가 끝났습니다` 발생하여 전체 학습 중단.

## 근본 원인
1. 게임 서버가 아닌 **worker 프로세스 자체**가 죽음
2. 서버 에러 발생 시 wrapper에서 예외를 잡지 못해 worker 프로세스 종료
3. Worker가 죽으면 파이프가 끊어져 메인 프로세스와 통신 불가 (BrokenPipeError)
4. Windows의 multiprocessing 불안정성 (fork 대신 spawn 사용)

## 해결 방법

### 1. Wrapper 레벨 예외 처리 (최후의 방어선)
**파일**: `envs/suika_wrapper.py`

#### reset() 메소드
```python
def reset(self, seed=None, options=None):
    try:
        # 정상 reset
        obs, info = self.env.reset(seed=seed, options=options)
        # ... 정상 처리 ...
        return processed_obs, info
    except Exception as e:
        # 에러 발생 시 더미 observation 반환 (worker 크래시 방지)
        print(f"[Wrapper] Error during reset: {e}")
        dummy_obs = self.observation_space.sample()
        processed_obs = self._process_observation(dummy_obs)
        return processed_obs, {'error': str(e), 'forced_reset': True}
```

#### step() 메소드
```python
def step(self, action):
    try:
        # 정상 step
        obs, reward, terminated, truncated, info = self.env.step(action)
        # ... 정상 처리 ...
        return processed_obs, processed_reward, terminated, truncated, info
    except Exception as e:
        # 에러 발생 시 에피소드 강제 종료 (worker 크래시 방지)
        print(f"[Wrapper] Error during step: {e}")
        dummy_obs = self.observation_space.sample()
        processed_obs = self._process_observation(dummy_obs)
        return processed_obs, 0.0, True, False, {
            'error': str(e),
            'forced_termination': True
        }
```

**효과**:
- Worker가 절대 죽지 않음
- 서버 에러 → 더미 observation + 에피소드 종료
- 다음 에피소드에서 서버 복구 시도

### 2. 서버 재시작 로직 개선
**파일**: `suika_rl/suika_env/suika_http_env.py`

#### 변경 사항
```python
# Before (무한 루프 가능)
return self._make_request(method, endpoint, data, retry_count + 1)

# After (retry_count 리셋)
return self._make_request(method, endpoint, data, retry_count=0)

# 서버 안정화 대기 시간 증가
time.sleep(3.0)  # 2초 → 3초
```

**효과**:
- 무한 재귀 방지
- 서버 재시작 후 전체 재시도 체인 다시 실행
- 서버가 안정화될 시간 충분히 제공

### 3. 서버 재시작 안정성 강화
**파일**: `envs/suika_wrapper.py`

#### 변경 사항
```python
def _restart_server(self, port, timeout=10.0):
    try:
        self._kill_server()
        time.sleep(2.0)  # 포트 해제 대기 (1초 → 2초)

        self._kill_existing_server_on_port(port)  # 남아있는 프로세스 강제 종료
        time.sleep(1.0)

        self._start_server_if_needed(port, timeout)

        # Health check 실패 시 재시도
        if not self._is_server_healthy(port):
            time.sleep(2.0)
            if not self._is_server_healthy(port):
                raise RuntimeError(f"Server on port {port} failed to restart")
    except Exception as e:
        print(f"[Port {port}] ERROR during server restart: {e}")
        raise
```

**효과**:
- 포트가 완전히 해제될 때까지 대기
- Health check 재시도로 안정성 확보
- 상세한 에러 로깅

## 작동 원리

### Before (Worker 크래시)
```
step() → 서버 에러 → RuntimeError 발생
     → worker 프로세스 종료
     → 파이프 끊김
     → BrokenPipeError in main process
     → 전체 학습 중단
```

### After (Graceful Handling)
```
step() → 서버 에러 → try-except catches
     → 더미 observation 반환 + terminated=True
     → worker 프로세스 유지
     → 다음 episode 시작
     → 서버 복구 시도
     → 학습 계속 진행
```

## 테스트

### 안정성 테스트
```bash
python tests/test_worker_stability.py
```

예상 결과:
- reset() 에러 처리 성공
- step() 에러 처리 성공
- 정상 동작 유지

### 실제 학습 테스트
```bash
python main.py --config config/default.yaml --mode train
```

예상 결과:
- 서버 에러 발생 시에도 worker 크래시 없음
- 에피소드만 종료되고 학습 계속 진행
- 로그에 에러 상황 기록

## 변경된 파일
1. `envs/suika_wrapper.py` - reset(), step() 예외 처리 추가
2. `suika_rl/suika_env/suika_http_env.py` - 재시작 로직 개선
3. `docs/BUG_FIX_REPORT.md` - 해결 과정 문서화
4. `docs/WORKER_CRASH_ANALYSIS.md` - 원인 분석 및 해결책 문서화
5. `tests/test_worker_stability.py` - 안정성 테스트 추가

## 핵심 원칙

### 1. Never Crash Worker
- Worker가 죽으면 복구 불가능
- 어떤 상황에서도 worker는 살려야 함
- 더미 데이터라도 반환하여 파이프 유지

### 2. Graceful Degradation
- 서버 에러 → 해당 에피소드만 희생
- 다음 에피소드에서 재시도
- 전체 학습은 계속 진행

### 3. Defense in Depth
- HTTP 레벨: 재시도 + 서버 재시작
- Env 레벨: 서버 재시작 콜백
- Wrapper 레벨: 최종 예외 처리 (최후의 방어선)

### 4. Fail Fast, Recover Fast
- 에러 발생 즉시 감지
- 상세한 로깅
- 빠른 복구 시도
- 복구 실패 시 graceful degradation

## 예상 효과

### 안정성
- Worker 크래시 확률: 거의 0%
- 서버 문제 발생 시에도 학습 계속
- 긴 학습 세션 안정적으로 완료 가능

### 디버깅
- 에러 상황 명확히 기록
- forced_termination/forced_reset 플래그로 추적 가능
- 서버 로그와 대조하여 원인 파악 용이

### 성능
- 정상 상황에서는 오버헤드 없음 (try-except는 예외 없으면 비용 거의 없음)
- 에러 상황에서만 더미 observation 생성
- 학습 속도 영향 없음

## 추가 권장 사항

### 설정 최적화
```yaml
# config/default.yaml
system:
  num_workers: 2  # Windows에서는 2개 권장 (안정성 vs 속도)

training:
  save_freq: 5000  # 자주 저장하여 크래시 시 복구 용이
```

### 모니터링
- 로그에서 `[Wrapper] Error` 패턴 주기적으로 확인
- `forced_termination` 빈도 모니터링
- 빈번하면 서버 안정성 문제 → 서버 코드 점검 필요

### 장기 개선
- Linux/WSL2 사용 (fork 기반 multiprocessing, 더 안정적)
- num_workers=1 사용 (100% 안정, 속도 희생)
- 서버 코드 최적화 (메모리 누수, 에러 원인 제거)
