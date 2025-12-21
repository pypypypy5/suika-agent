# 빠른 시작 가이드

수박게임 강화학습 프로젝트를 빠르게 시작하기 위한 가이드입니다.

## ⚡ 핵심 변경사항

**중요:** 이미지 기반 환경 대신 **상태 기반 환경**을 사용하세요!

- ✅ 1000배 효율적
- ✅ CNN 불필요, MLP만으로 OK
- ✅ 100배 빠른 학습
- ✅ 명확한 디버깅

```python
# ⭐ 추천: 상태 기반
from envs import SuikaStateWrapper
env = SuikaStateWrapper()
obs = np.ndarray(62,)  # 62개 값

# 호환성: 이미지 기반
from envs import SuikaEnvWrapper
env = SuikaEnvWrapper()
obs = {'image': (400,300,3), 'score': float}
```

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

### Suika RL 환경 설치
```bash
# 프로젝트 루트에서
git clone https://github.com/edwhu/suika_rl.git
cd suika_rl
pip install -e .
cd ..
```

## 2. 예제 실행 (1분)

환경이 제대로 설정되었는지 확인:
```bash
python example_usage.py
```

이 예제는 다음을 보여줍니다:
- 환경과의 기본 상호작용
- 에이전트 사용법
- 커스텀 보상 함수
- 통계 정보 활용

## 3. 첫 번째 에이전트 구현

### 3.1 에이전트 파일 생성

`agents/my_agent.py` 파일을 만들고 다음 템플릿을 사용:

```python
import torch
import torch.nn as nn
from agents.base_agent import RLAgent

class MyMLPAgent(RLAgent):
    """상태 기반 환경용 MLP 에이전트"""

    def __init__(self, observation_space, action_space, config=None):
        super().__init__(observation_space, action_space, config)

        # MLP 네트워크 (CNN 필요 없음!)
        obs_dim = observation_space.shape[0]  # 62
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0~1 범위 행동
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

    def _forward_policy(self, obs, deterministic):
        # 간단한 forward pass
        return self.policy_net(obs)

    def update(self, obs, action, reward, next_obs, done):
        # 학습 로직 (DQN, PPO 등)
        # return {'loss': loss_value}
        pass
```

### 3.2 main.py에 에이전트 등록

`main.py`의 `create_agent` 함수에 추가:

```python
elif agent_type == 'my_agent':
    from agents.my_agent import MyAgent
    agent = MyAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=agent_config
    )
```

### 3.3 설정 파일 수정

`config/default.yaml`에서 에이전트 타입 변경:

```yaml
agent:
  type: "my_agent"  # 'random'에서 'my_agent'로 변경
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

## 추천 학습 순서

### Phase 1: 환경 이해
1. `example_usage.py` 실행하여 환경 파악
2. `envs/suika_wrapper.py` 코드 읽기
3. 관찰 공간과 행동 공간 이해

### Phase 2: 베이스라인 설정
1. Random Agent로 학습 실행
2. 성능 기록 (평균 보상 등)
3. 이것이 개선 목표!

### Phase 3: 알고리즘 구현
1. 간단한 알고리즘부터 시작 (DQN 추천)
2. `agents/base_agent.py`의 `RLAgent` 상속
3. `_forward_policy`와 `update` 메서드 구현

### Phase 4: 하이퍼파라미터 튜닝
1. `config/default.yaml`에서 파라미터 조정
2. 학습률, 배치 크기, 네트워크 구조 실험
3. WandB로 실험 추적 (선택사항)

## 프로젝트 구조 요약

```
melon-ai/
├── envs/              # 환경 래퍼 (수정 불필요)
│   └── suika_wrapper.py
├── agents/            # 여기에 에이전트 구현!
│   ├── base_agent.py  # 상속받을 베이스 클래스
│   └── your_agent.py  # 구현할 에이전트
├── models/            # 신경망 모델 (선택사항)
├── training/          # 학습 루프 (수정 불필요)
├── utils/             # 유틸리티 (수정 불필요)
├── config/            # 설정 파일
│   └── default.yaml   # 하이퍼파라미터 조정
└── main.py           # 실행 파일
```

## 핵심 인터페이스

에이전트 구현 시 필수 메서드:

```python
class YourAgent(RLAgent):
    def _forward_policy(self, obs, deterministic):
        """관찰 -> 행동"""

    def update(self, obs, action, reward, next_obs, done):
        """경험으로부터 학습"""

    def save(self, path):
        """모델 저장"""

    def load(self, path):
        """모델 로드"""
```

환경 인터페이스는 표준 Gymnasium:
- `env.reset()` → observation, info
- `env.step(action)` → observation, reward, terminated, truncated, info

## 문제 해결

### Suika RL 환경이 없는 경우
- Mock 환경이 자동으로 사용됩니다
- 실제 환경 설치는 섹션 1 참고

### CUDA 메모리 부족
- `config/default.yaml`에서 `batch_size` 줄이기
- 네트워크 크기 축소

### 학습이 너무 느린 경우
- `system.num_workers` 증가 (병렬 환경)
- 평가 빈도 줄이기 (`training.eval_freq`)

## 다음 단계

1. DQN 구현 예제 찾기
2. PPO, SAC 등 다른 알고리즘 시도
3. 커스텀 보상 함수 설계
4. 네트워크 아키텍처 개선

행운을 빕니다! 🍉
