# Performance Testing

이 디렉토리는 수박게임 RL 환경의 성능 측정 및 프로파일링 도구를 포함합니다.

## 파일 설명

### 성능 측정 스크립트

- **`profile_env_performance.py`** - 전체 성능 측정
  - 여러 에피소드에 걸쳐 평균 스텝 시간, 처리량, 후반부 느려짐 등을 측정
  - 가장 기본적인 성능 벤치마크

- **`profile_detailed.py`** - 병목 지점 상세 분석
  - 스텝 내부의 각 작업(액션 전송, 화면 캡처, 물리 대기 등)별 시간 측정
  - 어느 부분이 가장 느린지 정확히 파악

- **`profile_late_episode.py`** - 후반부 성능 저하 조사
  - 에피소드 진행에 따른 성능 변화 추적
  - 물리 객체 수와 스텝 시간의 상관관계 분석

- **`profile_final.py`** - 최종 성능 측정 및 비교
  - 최적화 전후 성능 비교
  - 포괄적인 통계 및 벤치마크 결과 생성

### 검증 스크립트

- **`test_optimizations.py`** - 최적화 기능 검증
  - 최적화가 기존 기능을 깨뜨리지 않았는지 확인
  - 기본 기능 및 fast mode 동작 테스트

## 사용법

### 기본 성능 측정
```bash
python tests/performance/profile_env_performance.py
```

### 상세 병목 분석
```bash
python tests/performance/profile_detailed.py
```

### 후반부 느려짐 조사
```bash
python tests/performance/profile_late_episode.py
```

### 최적화 전후 비교
```bash
python tests/performance/profile_final.py
```

### 기능 검증
```bash
python tests/performance/test_optimizations.py
```

## 성능 최적화 결과

자세한 최적화 결과는 [`docs/OPTIMIZATION_REPORT.md`](../../docs/OPTIMIZATION_REPORT.md) 참조.

**요약:**
- 속도 향상: **2.51배** (60.1% 빠름)
- 처리량: 2.77 → 6.94 steps/s
- 학습 시간: 5시간 → 2시간 (1000 에피소드 기준)
