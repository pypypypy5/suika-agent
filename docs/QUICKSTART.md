# 빠른 시작 가이드

수박게임 강화학습 프로젝트를 빠르게 시작하기 위한 가이드입니다.

## ⚡ 환경 개요

이 프로젝트는 **이미지 기반 환경**을 사용합니다:

```python
from envs import SuikaEnvWrapper

env = SuikaEnvWrapper()
obs, info = env.reset()
# obs = {'image': (400, 300, 3), 'score': float}
```

**필요한 모델:** CNN (Convolutional Neural Network)

## 1. 환경 설정 (5분)

### 가상환경 생성 및 활성화
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 의존성 설치
```bash
pip install -r requirements.txt
```

**주의**: `suika_rl`은 이미 프로젝트에 포함되어 있습니다. 별도 설치 불필요!

## 2. 예제 실행 (1분)

환경이 제대로 설정되었는지 확인:
```bash
python example_usage.py
```

또는 테스트 실행:
```bash
python tests/test_simple.py
```

## 3. 첫 번째 CNN 에이전트 구현

### 3.1 에이전트 파일 생성

`agents/cnn_agent.py` 파일을 만들고 다음 템플릿을 사용:

```python
import torch
import torch.nn as nn
from agents.base_agent import RLAgent

class CNNAgent(RLAgent):
    """이미지 기반 환경용 CNN 에이전트"""

    def __init__(self, observation_space, action_space, config=None):
        super().__init__(observation_space, action_space, config)

        # CNN 네트워크 (이미지 처리)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(64 * 46 * 34, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 0~1 범위 행동
        )

        self.policy_net = nn.Sequential(self.conv, nn.Flatten(), self.fc).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

    def _forward_policy(self, obs, deterministic):
        # 이미지 전처리 및 forward pass
        return self.policy_net(obs)

    def update(self, obs, action, reward, next_obs, done):
        # 학습 로직 (DQN, Rainbow 등)
        # return {'loss': loss_value}
        pass
```

### 3.2 main.py에 에이전트 등록

`main.py`의 `create_agent` 함수에 추가:

```python
elif agent_type == 'cnn':
    from agents.cnn_agent import CNNAgent
    agent = CNNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=agent_config
    )
```

### 3.3 설정 파일 수정

`config/default.yaml`에서 에이전트 타입 변경:

```yaml
agent:
  type: "cnn"  # 'random'에서 'cnn'로 변경
```

## 4. 학습 시작

```bash
python main.py --mode train --config config/default.yaml
```

학습 진행 상황은 다음에서 확인:
- 콘솔 출력
- TensorBoard: `tensorboard --logdir experiments/tensorboard`
- 체크포인트: `experiments/checkpoints/`

## 5. 모델 평가

```bash
python main.py --mode eval --checkpoint experiments/checkpoints/best_model.pth
```

## 권장 알고리즘

이미지 기반 환경에 적합한 알고리즘:

### 1. DQN (Deep Q-Network)
- CNN으로 이미지 처리
- Q-learning + Experience Replay
- 구현 난이도: ⭐⭐⭐

### 2. Rainbow DQN
- DQN + 6가지 개선사항
- 최신 기법 집합
- 구현 난이도: ⭐⭐⭐⭐

### 3. PPO (with CNN)
- Policy Gradient 기반
- 안정적인 학습
- 구현 난이도: ⭐⭐⭐⭐

### 4. Stable Baselines3 사용
```python
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: SuikaEnvWrapper()])
model = DQN('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

## 프로젝트 구조 요약

```
melon-ai/
├── envs/              # 환경 래퍼
│   └── suika_wrapper.py  # 이미지 기반
├── agents/            # 여기에 CNN 에이전트 구현!
│   ├── base_agent.py
│   └── cnn_agent.py  # 구현할 파일
├── models/            # CNN 아키텍처 (선택사항)
├── training/          # 학습 루프
├── config/            # 설정 파일
│   └── default.yaml   # 하이퍼파라미터 조정
└── main.py           # 실행 파일
```

## 문제 해결

### Chrome/Chromium 설치
```bash
# Ubuntu/Debian
sudo apt-get install chromium-browser chromium-chromedriver

# macOS
brew install chromedriver
```

### CUDA 메모리 부족
- `config/default.yaml`에서 `batch_size` 줄이기
- 이미지 크기 축소

### 학습이 너무 느린 경우
- GPU 사용 확인
- 배치 크기 증가
- 병렬 환경 사용 (`system.num_workers`)

## 다음 단계

1. **CNN 구조 개선**: 더 깊은 네트워크, Residual connection 등
2. **알고리즘 적용**: DQN, Rainbow, PPO 등
3. **하이퍼파라미터 튜닝**: 학습률, 배치 크기 등

행운을 빕니다! 🍉
