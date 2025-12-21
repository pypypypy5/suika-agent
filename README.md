# Suika Game Reinforcement Learning Project

수박게임(Suika Game)을 플레이하는 강화학습 에이전트를 학습시키기 위한 프로젝트입니다.

## 🎯 주요 특징

- ✅ **효율적인 상태 기반 환경**: 이미지 대신 구조화된 게임 상태 사용 (1000배 효율 개선)
- ✅ **실제 게임 통합**: JavaScript 수박게임 + Selenium WebDriver
- ✅ **Best Practice 준수**: Gymnasium 표준, 모듈화, 설정 기반 관리
- ✅ **완전한 추상화**: 에이전트는 환경 세부사항을 몰라도 됨
- ✅ **두 가지 환경 제공**: 이미지 기반(호환성) + 상태 기반(추천)

## 프로젝트 구조

```
melon-ai/
├── suika_rl/              # 외부 오픈소스 (포함됨)
│   └── suika_env/
│       ├── suika-game/    # JavaScript 게임 (수정됨: label 추가)
│       └── suika_browser_env.py  # Selenium 래퍼
├── config/                # 설정 파일
│   └── default.yaml       # 하이퍼파라미터
├── envs/                  # 환경 래퍼 (핵심!)
│   ├── suika_wrapper.py         # 이미지 기반 (호환성)
│   └── suika_state_wrapper.py   # ⭐ 상태 기반 (추천)
├── agents/                # 에이전트 구현 (사용자가 추가)
│   └── base_agent.py      # 베이스 클래스
├── models/                # 신경망 모델
├── training/              # 학습 프레임워크
│   └── trainer.py
├── utils/                 # 유틸리티
│   └── logger.py
├── tests/                 # 테스트 코드
│   ├── test_simple.py
│   ├── test_environment_api.py
│   └── test_state_env.py
├── experiments/           # 실험 결과
├── venv/                  # 가상환경
├── requirements.txt       # 의존성
└── main.py               # 메인 실행
```

## 설치 방법

### 1. 가상환경 생성
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. (선택사항) Chrome/Chromium 설치
실제 Suika 게임 환경을 사용하려면 Chrome과 ChromeDriver가 필요합니다:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install chromium-browser chromium-chromedriver

# macOS
brew install chromedriver

# Windows
# https://chromedriver.chromium.org/ 에서 다운로드
```

**주의**: Chrome 없이도 Mock 환경으로 개발/테스트가 가능합니다.

### 4. 통합 테스트
```bash
python test_integration.py
```

## 사용 방법

### API 테스트 (먼저 실행 권장)
```bash
# Linux/Mac
bash run_api_test.sh

# Windows
run_api_test.bat

# 또는 직접
python tests/test_simple.py
```

### 학습 시작
```bash
python main.py --mode train --config config/default.yaml
```

### 학습된 모델 평가
```bash
python main.py --mode eval --checkpoint experiments/checkpoints/best_model.pth
```

## 두 가지 환경 비교

### 1. 상태 기반 환경 (추천!) ⭐

```python
from envs import SuikaStateWrapper

env = SuikaStateWrapper(headless=True, max_fruits=20)
obs, info = env.reset()

# obs = np.ndarray(62,)
# [next_fruit, score, fruit1_x, fruit1_y, fruit1_type, ...]
```

**장점:**
- ✅ **관찰 크기**: 62개 값 (vs 이미지 65,536개)
- ✅ **메모리**: 0.24 KB (vs 이미지 256 KB)
- ✅ **필요 모델**: MLP (vs CNN)
- ✅ **학습 속도**: 100배 이상 빠름
- ✅ **해석 가능**: 각 값의 의미가 명확

### 2. 이미지 기반 환경 (호환성)

```python
from envs import SuikaEnvWrapper

env = SuikaEnvWrapper(headless=True)
obs, info = env.reset()

# obs = {'image': (400, 300, 3), 'score': float}
```

**용도**: CNN 기반 알고리즘 연구 또는 기존 코드 호환성

---

## 환경 API

에이전트는 환경의 세부사항을 몰라도 됩니다:

- **관찰(Observation)**: 게임 상태 (상태 기반) 또는 이미지 + 점수
- **행동(Action)**: 과일을 떨어뜨릴 위치 [0.0 ~ 1.0]
- **보상(Reward)**: 점수 증가량
- **종료(Done)**: terminated (게임 오버), truncated (시간 제한)

## 추가 문서

- **ARCHITECTURE.md** - 프로젝트 아키텍처 상세 설명
- **STATE_BASED_IMPROVEMENT.md** - 상태 기반 환경 개선 내용
- **TEST_RESULTS.md** - 테스트 결과 및 API 검증
- **FINAL_REPORT.md** - 최종 완성 보고서
- **QUICKSTART.md** - 빠른 시작 가이드

## 참고 자료

### 사용된 오픈소스
- [TomboFry/suika-game](https://github.com/TomboFry/suika-game) - JavaScript 게임 (수정됨)
- [edwhu/suika_rl](https://github.com/edwhu/suika_rl) - RL 환경 베이스

### RL 프레임워크
- [Gymnasium](https://gymnasium.farama.org/) - 표준 RL 인터페이스
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL 알고리즘

### Best Practices
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Deep RL Course](https://huggingface.co/learn/deep-rl-course/)
