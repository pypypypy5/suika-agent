# Suika Game Reinforcement Learning Project

수박게임(Suika Game)을 플레이하는 강화학습 에이전트를 학습시키기 위한 프로젝트입니다.

## 🎯 주요 특징

- ✅ **이미지 기반 환경**: CNN으로 게임 화면을 직접 처리
- ✅ **실제 게임 통합**: JavaScript 수박게임 + Selenium WebDriver
- ✅ **Best Practice 준수**: Gymnasium 표준, 모듈화, 설정 기반 관리
- ✅ **완전한 추상화**: 에이전트는 환경 세부사항을 몰라도 됨
- ✅ **Deep RL 지원**: DQN, Rainbow 등 이미지 기반 알고리즘 적용 가능

## 프로젝트 구조

```
melon-ai/
├── suika_rl/              # 외부 오픈소스 (포함됨)
│   └── suika_env/
│       ├── suika-game/    # JavaScript 게임
│       └── suika_browser_env.py  # Selenium 래퍼
├── config/                # 설정 파일
│   └── default.yaml       # 하이퍼파라미터
├── envs/                  # 환경 래퍼
│   └── suika_wrapper.py   # 이미지 기반 환경
├── agents/                # 에이전트 구현 (사용자가 추가)
│   └── base_agent.py      # 베이스 클래스
├── models/                # 신경망 모델 (CNN)
├── training/              # 학습 프레임워크
│   └── trainer.py
├── utils/                 # 유틸리티
│   └── logger.py
├── tests/                 # 테스트 코드
│   ├── test_simple.py
│   └── test_environment_api.py
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

### 이전 모델에서 이어서 학습
```bash
python main.py --mode train --config config/default.yaml --resume experiments/checkpoints/DQNAgent_0126_2038_best.pth
```

### 학습된 모델 평가
```bash
python main.py --mode eval --checkpoint experiments/checkpoints/best_model.pth
```

## 환경 사용법

```python
from envs import SuikaEnvWrapper

# 환경 생성
env = SuikaEnvWrapper(headless=True, normalize_obs=True)
obs, info = env.reset()

# obs = {'image': (400, 300, 3), 'score': float}
# 이미지는 자동으로 [0, 1] 범위로 정규화됨
```

**관찰:**
- **image**: 게임 화면 (400, 300, 3) - RGB 이미지
- **score**: 현재 점수

**행동:**
- 과일을 떨어뜨릴 위치 [0.0 ~ 1.0]

**보상:**
- 점수 증가량

**종료:**
- terminated (게임 오버)
- truncated (시간 제한)

---

## CNN 에이전트 예시

이미지 기반 환경이므로 CNN을 사용한 에이전트가 필요합니다:

```python
import torch.nn as nn

class CNNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN으로 이미지 처리
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(64 * 46 * 34, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

## 추가 문서

모든 개발 문서는 `docs/` 디렉토리에 정리되어 있습니다:

- **docs/ARCHITECTURE.md** - 프로젝트 아키텍처 상세 설명
- **docs/QUICKSTART.md** - 빠른 시작 가이드
- **docs/TEST_RESULTS.md** - 테스트 결과 및 API 검증

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
