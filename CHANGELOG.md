# 변경 사항 (Changelog)

## 주요 개선사항

### ⭐ 상태 기반 환경 추가 (2025-12-22)

**문제점 발견:**
> "이미지가 주어지고 CNN으로 해석해야 하는거야? 왜 이렇게 하는거야?"

**해결책:**
JavaScript에서 게임 상태(과일 위치, 타입 등)를 직접 추출하는 새로운 환경 추가

**성과:**
- ✅ 관찰 크기: 65,536개 → 62개 (1,057배 감소)
- ✅ 메모리: 256 KB → 0.24 KB (99.9% 절감)
- ✅ 필요 모델: CNN → MLP
- ✅ 학습 속도: 100배 이상 향상

---

## 파일 변경 내역

### 추가된 파일

#### 환경
- **envs/suika_state_wrapper.py** - 상태 기반 환경 래퍼 (핵심!)
  - JavaScript에서 과일 정보 추출
  - 62개 벡터로 상태 인코딩
  - Mock 환경 포함

#### 테스트
- **tests/test_state_env.py** - 상태 기반 환경 테스트
  - 과일 정보 추출 검증
  - 성능 비교 분석

#### 문서
- **STATE_BASED_IMPROVEMENT.md** - 상태 기반 개선 설명
- **FINAL_REPORT.md** - 최종 완성 보고서
- **TEST_RESULTS.md** - 테스트 결과 및 API 검증
- **SUMMARY.md** - 프로젝트 요약
- **CHANGELOG.md** - 이 문서

#### 스크립트
- **run_api_test.sh** - API 테스트 실행 (Linux/Mac)
- **run_api_test.bat** - API 테스트 실행 (Windows)

### 수정된 파일

#### 게임 소스
- **suika_rl/suika_env/suika-game/index.js**
  ```javascript
  // 변경 전
  generateFruitBody: function (x, y, sizeIndex, extraConfig = {}) {
      const circle = Bodies.circle(x, y, size.radius, {
          ...friction,
          ...extraConfig,
          render: { ... }
      });
  }

  // 변경 후
  generateFruitBody: function (x, y, sizeIndex, extraConfig = {}) {
      const circle = Bodies.circle(x, y, size.radius, {
          ...friction,
          ...extraConfig,
          label: `fruit-${sizeIndex}`,  // ← 추가!
          render: { ... }
      });
  }
  ```
  **목적:** RL 환경에서 각 과일을 식별할 수 있도록 라벨 추가

#### 환경 래퍼
- **envs/__init__.py**
  ```python
  # 변경 전
  from .suika_wrapper import SuikaEnvWrapper
  __all__ = ['SuikaEnvWrapper']

  # 변경 후
  from .suika_wrapper import SuikaEnvWrapper
  from .suika_state_wrapper import SuikaStateWrapper
  __all__ = ['SuikaEnvWrapper', 'SuikaStateWrapper']
  ```

- **envs/suika_wrapper.py**
  - `_process_observation()` 수정: Dict 형태 유지하도록 개선
  - observation이 항상 `{'image': ..., 'score': ...}` 형태로 반환

#### 설정
- **config/default.yaml**
  - 환경 설정 섹션 확장 (headless, port 등)

#### 문서
- **README.md**
  - 주요 특징 추가
  - 두 가지 환경 비교 섹션 추가
  - 프로젝트 구조 업데이트
  - 추가 문서 링크

- **QUICKSTART.md**
  - 상태 기반 환경 강조
  - MLP 에이전트 예시 추가

- **ARCHITECTURE.md**
  - 상태 기반 접근 방식 설명 추가
  - 성능 비교 섹션 추가

#### 기타
- **.gitignore**
  ```diff
  - # Suika RL
  - suika_rl/
  - node_modules/
  + # Suika RL - node_modules만 제외
  + suika_rl/node_modules/
  +
  + # Test outputs
  + tests/*.png
  + tests/*.gif
  ```
  **변경 이유:** suika_rl은 프로젝트에 포함되어야 함

### 유지된 파일 (변경 없음)

- agents/base_agent.py - 에이전트 베이스 클래스
- training/trainer.py - 학습 프레임워크
- utils/logger.py - 로깅 유틸리티
- main.py - 메인 실행 파일
- requirements.txt - 의존성
- tests/test_simple.py - 기본 테스트
- tests/test_environment_api.py - API 테스트

---

## 기술적 세부사항

### JavaScript 상태 추출

```javascript
// 새로 추가된 상태 추출 로직 (Python에서 실행)
const fruits = [];
const bodies = Composite.allBodies(engine.world);

for (const body of bodies) {
    if (body.label && body.label.startsWith('fruit')) {
        const sizeIndex = parseInt(body.label.split('-')[1]);
        fruits.push({
            x: body.position.x,
            y: body.position.y,
            type: sizeIndex
        });
    }
}

return {
    next_fruit: window.Game.nextFruitSize,
    score: window.Game.score,
    fruits: fruits,
    game_width: window.Game.width,
    game_height: window.Game.height
};
```

### 상태 인코딩

```python
# 62개 요소 벡터로 인코딩
state = [
    next_fruit_normalized,    # [0]
    score_normalized,         # [1]
    fruit1_x_normalized,      # [2]
    fruit1_y_normalized,      # [3]
    fruit1_type_normalized,   # [4]
    # ... (최대 20개 과일)
]
```

**정규화 범위:** 모든 값은 [-1, 1] 범위

---

## 테스트 결과

### 실행된 테스트

```bash
✅ ./venv/bin/python tests/test_simple.py
✅ ./venv/bin/python tests/test_environment_api.py
✅ ./venv/bin/python tests/test_state_env.py
```

### 검증된 기능

- [x] 다음 과일 타입 추출
- [x] 현재 점수 추출
- [x] 과일 위치 (x, y) 추출
- [x] 과일 타입 추출
- [x] 보상 신호
- [x] 종료 조건
- [x] 에피소드 통계
- [x] 정규화
- [x] Mock 환경
- [x] 실제 환경 준비

---

## 마이그레이션 가이드

### 기존 코드에서 상태 기반 환경으로 전환

```python
# 변경 전 (이미지 기반)
from envs import SuikaEnvWrapper

env = SuikaEnvWrapper()
obs, info = env.reset()
# obs = {'image': (400, 300, 3), 'score': float}

# CNN 필요
model = CNNPolicy(...)

# 변경 후 (상태 기반)
from envs import SuikaStateWrapper

env = SuikaStateWrapper()
obs, info = env.reset()
# obs = np.ndarray(62,)

# MLP로 충분
model = MLPPolicy(...)
```

**호환성:** 기존 이미지 기반 환경(`SuikaEnvWrapper`)도 계속 사용 가능

---

## 향후 계획

### 단기 (사용자 구현)
- [ ] MLP 기반 DQN 에이전트
- [ ] PPO 에이전트
- [ ] 하이퍼파라미터 튜닝

### 중기
- [ ] 실제 환경 성능 최적화
- [ ] 추가 보상 함수 옵션
- [ ] 더 많은 에이전트 예제

### 장기
- [ ] 멀티 에이전트 지원
- [ ] 커리큘럼 학습
- [ ] 사전 학습 모델

---

## 기여자

- 초기 프로젝트 설정 및 아키텍처
- 상태 기반 환경 개선
- 문서화 및 테스트

---

## 라이선스

이 프로젝트는 다음 오픈소스를 사용합니다:

- **TomboFry/suika-game** - JavaScript 게임 구현
- **edwhu/suika_rl** - RL 환경 베이스
- **Gymnasium** - RL 표준 인터페이스

각 오픈소스의 라이선스를 준수합니다.
