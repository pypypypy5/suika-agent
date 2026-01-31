# 메모리 안정성 문제 해결 보고서

## 문제 상황
학습 실행 시 수만 step 이후 **파이썬 env 프로세스가 메모리 부족으로 종료**되는 문제 발생

### 구조
```
Trainer → VectorEnv → [Env1, Env2, ...] (파이썬 프로세스)
                           ↓
                    SuikaBrowserEnv (HTTP)
                           ↓
                    Node.js 게임 서버
```

## 해결 방법

### 1. 메모리 누수 수정 (`suika_rl/suika_env/suika_http_env.py`)
```python
# PIL Image 명시적 close 추가
img = Image.open(io.BytesIO(image_string))
arr = np.asarray(img).copy()
img.close()  # 추가!
```

### 2. 주기적 가비지 컬렉션
- 1000 step마다: `gc.collect()`
- 100 에피소드마다: `gc.collect()`

### 3. 메모리 모니터링 로그
```python
import psutil
process = psutil.Process()
current_memory = process.memory_info().rss / 1024 / 1024  # MB
logger.info(f"Step {step_count}, Memory: {current_memory:.2f} MB")
```

### 4. Chrome 옵션 강화 (`suika_browser_env.py`)
```python
opts.add_argument("--disable-gpu")
opts.add_argument("--disable-software-rasterizer")
```

## 검증 결과

### 5000 steps 테스트 (2026-01-31)
- ✅ **크래시 없음**
- ✅ **메모리 안정** (200MB → 135MB로 오히려 감소)
- ✅ **학습 완료**

```
초기:   ~200 MB
2000s:  ~190 MB
2600s:  ~135 MB  ← 메모리 감소!
```

## 다음 작업

### 장기 안정성 검증 필요
```bash
# 50,000+ steps 테스트
python main.py --mode train --config config/long_training.yaml
```

### 모니터링 포인트
- 메모리 로그: `grep "Memory:" logs/*.log`
- 프로세스 상태: VectorEnv worker 프로세스 확인
- 에러: `grep "ERROR\|Exception" logs/*.log`

## 변경 파일
1. `suika_rl/suika_env/suika_http_env.py` - 메모리 관리 로그 + GC
2. `suika_rl/suika_env/suika_browser_env.py` - PIL close + Chrome 옵션 + 로그
3. `config/test_stability.yaml` - 짧은 테스트용 설정

## 참고
- venv 사용: `./venv/Scripts/python.exe`
- 서버 자동 시작: wrapper가 자동으로 Node.js 서버 실행
- 로그 위치: `logs/server_port{PORT}_*.log`
