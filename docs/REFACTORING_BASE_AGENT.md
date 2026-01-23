# Base Agent 리팩터링 인수인계 문서

## 현재 상황 요약

### 문제점
Base Agent와 Simple Agent 간에 전처리 로직이 중복 구현되어 있음. Simple Agent가 Base Agent의 `preprocess_observation()` 메서드를 사용하지 않고 자체적으로 `_extract_observation()`과 `_preprocess_observation()`을 구현하고 있음.

### 왜 문제인가?
1. **코드 중복**: 동일한 로직이 여러 곳에 존재하여 유지보수 어려움
2. **일관성 부족**: Base Agent의 메서드가 있지만 실제로는 사용되지 않음
3. **확장성 저해**: 새로운 Agent(예: DQN) 구현 시 전처리 로직을 또 다시 복사해야 함

---

## Base Agent의 현재 구현 (base_agent.py:187-212)

```python
def preprocess_observation(self, obs: Union[np.ndarray, Dict]) -> torch.Tensor:
    """관찰 배치를 신경망 입력으로 전처리"""
    if isinstance(obs, dict):
        # 딕셔너리 형태의 관찰 처리
        obs = obs.get('image', list(obs.values())[0])  # ← 문제: obs_key 고려 안함

    # NumPy 배열을 텐서로 변환
    if not isinstance(obs, torch.Tensor):
        obs = torch.FloatTensor(obs)

    # 이미지면 채널을 앞으로 (PyTorch 컨벤션)
    if obs.dim() == 4 and obs.shape[-1] in [1, 3, 4]:  # (N, H, W, C)
        obs = obs.permute(0, 3, 1, 2)  # (N, C, H, W)

    return obs.to(self.device)
```

**문제점:**
- `obs_key`를 설정에서 받아서 처리하지 않음 (하드코딩: `'image'`)
- `is_dict_obs` 같은 플래그가 없어서 매번 타입 체크 필요

---

## Simple Agent의 올바른 구현 (simple_agent.py)

### 1. __init__에서 observation space 분석 (144-172줄)

```python
from gymnasium import spaces as gym_spaces

if isinstance(observation_space, gym_spaces.Dict):
    # Dict observation space - 'image' 키 사용
    self.is_dict_obs = True
    self.obs_key = config.get('obs_key', 'image')  # ← config에서 받음

    if self.obs_key not in observation_space.spaces:
        raise ValueError(
            f"observation_space에 '{self.obs_key}' 키가 없습니다. "
            f"사용 가능한 키: {list(observation_space.spaces.keys())}"
        )

    raw_obs_shape = observation_space.spaces[self.obs_key].shape
else:
    # 단일 observation space
    self.is_dict_obs = False
    self.obs_key = None
    raw_obs_shape = observation_space.shape

# 이미지 입력이면 (H, W, C) -> (C, H, W)로 변환
if len(raw_obs_shape) == 3:
    # 이미지로 가정: (H, W, C) -> (C, H, W)
    num_channels = raw_obs_shape[2]
    self.obs_shape = (num_channels, raw_obs_shape[0], raw_obs_shape[1])
else:
    # 벡터 입력
    self.obs_shape = raw_obs_shape
```

### 2. Dict에서 관찰 추출 (197-218줄)

```python
def _extract_observation(self, observation: Union[np.ndarray, Dict]) -> np.ndarray:
    """Dict observation에서 실제 관찰 추출 (NumPy 유지)"""
    if isinstance(observation, dict):
        if self.obs_key and self.obs_key in observation:
            obs = observation[self.obs_key]  # ← self.obs_key 활용
        else:
            # 첫 번째 값 사용
            obs = list(observation.values())[0]
    else:
        obs = observation

    return obs
```

### 3. NumPy를 Tensor로 변환 및 전처리 (220-240줄)

```python
def _preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
    """관찰 배치를 신경망 입력으로 전처리"""
    # NumPy to Tensor
    obs_tensor = torch.FloatTensor(obs).to(self.device)

    # 이미지면 전처리
    if len(obs_tensor.shape) == 4:  # (N, H, W, C)
        # (N, H, W, C) -> (N, C, H, W)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        # NOTE: 정규화는 wrapper에서 이미 완료되었으므로 여기서는 하지 않음

    return obs_tensor
```

### 4. select_action에서 사용 (242-257줄)

```python
def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False):
    # 관찰 추출 및 전처리
    obs = self._extract_observation(observation)  # Dict → NumPy
    obs_tensor = self._preprocess_observation(obs)  # NumPy → Tensor

    # 네트워크 forward
    with torch.no_grad():
        logits = self.policy_net(obs_tensor)
        # ...
```

---

## 리팩터링 계획

### Step 1: Base Agent (RLAgent) 개선

`agents/base_agent.py`의 `RLAgent.__init__()`에 다음 추가:

```python
def __init__(self, observation_space, action_space, config=None, device=None):
    super().__init__(observation_space, action_space, config)

    # 디바이스 설정
    if device is None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        self.device = torch.device(device)

    # 학습 파라미터
    self.gamma = config.get('gamma', 0.99)
    self.learning_rate = config.get('learning_rate', 3e-4)
    self.batch_size = config.get('batch_size', 64)

    # ========== 새로 추가할 부분 ==========
    from gymnasium import spaces as gym_spaces

    # Observation space 분석
    if isinstance(observation_space, gym_spaces.Dict):
        self.is_dict_obs = True
        self.obs_key = config.get('obs_key', 'image')

        if self.obs_key not in observation_space.spaces:
            raise ValueError(
                f"observation_space에 '{self.obs_key}' 키가 없습니다. "
                f"사용 가능한 키: {list(observation_space.spaces.keys())}"
            )

        raw_obs_shape = observation_space.spaces[self.obs_key].shape
    else:
        self.is_dict_obs = False
        self.obs_key = None
        raw_obs_shape = observation_space.shape

    # Observation shape 계산 (PyTorch 형식: C, H, W)
    if len(raw_obs_shape) == 3:
        # 이미지: (H, W, C) -> (C, H, W)
        num_channels = raw_obs_shape[2]
        self.obs_shape = (num_channels, raw_obs_shape[0], raw_obs_shape[1])
    else:
        # 벡터 입력
        self.obs_shape = raw_obs_shape
    # ========================================

    # 신경망 모델 (하위 클래스에서 초기화)
    self.policy_net: Optional[nn.Module] = None
    self.optimizer: Optional[torch.optim.Optimizer] = None
```

### Step 2: Base Agent에 전처리 메서드 추가/개선

기존 `preprocess_observation()` 메서드를 다음과 같이 교체:

```python
def extract_observation(self, observation: Union[np.ndarray, Dict]) -> np.ndarray:
    """
    Dict observation에서 실제 관찰 추출 (NumPy 유지)

    Args:
        observation: 환경의 관찰 (Dict 또는 array)
            - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
            - Array: (N, ...)

    Returns:
        추출된 관찰 array (N, H, W, C) 또는 (N, ...)
    """
    if self.is_dict_obs and isinstance(observation, dict):
        if self.obs_key and self.obs_key in observation:
            return observation[self.obs_key]
        else:
            # Fallback: 첫 번째 값 사용
            return list(observation.values())[0]
    return observation


def preprocess_observation(self, obs: Union[np.ndarray, Dict]) -> torch.Tensor:
    """
    관찰을 신경망 입력으로 전처리 (Dict → NumPy → Tensor)

    Args:
        obs: 원본 관찰 배치
            - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
            - Array: (N, H, W, C) 또는 (N, features)

    Returns:
        전처리된 텐서 (N, C, H, W) 또는 (N, features)
    """
    # 1. Dict에서 추출
    obs_array = self.extract_observation(obs)

    # 2. NumPy to Tensor
    obs_tensor = torch.FloatTensor(obs_array).to(self.device)

    # 3. 이미지면 채널 순서 변경
    if len(obs_tensor.shape) == 4:  # (N, H, W, C)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)

    return obs_tensor
```

### Step 3: Simple Agent 리팩터링

`agents/simple_agent.py` 수정:

1. **__init__에서 observation space 분석 코드 제거** (144-172줄 삭제)
   - Base Agent가 이미 처리하므로 불필요

2. **_extract_observation, _preprocess_observation 메서드 삭제** (197-240줄 삭제)
   - Base Agent의 메서드 사용

3. **select_action 수정** (242-257줄):
   ```python
   def select_action(self, observation, deterministic=False):
       # 기존:
       # obs = self._extract_observation(observation)
       # obs_tensor = self._preprocess_observation(obs)

       # 변경:
       obs_tensor = self.preprocess_observation(observation)  # Base의 메서드 사용

       # 네트워크 forward (동일)
       with torch.no_grad():
           logits = self.policy_net(obs_tensor)
           # ...
   ```

4. **store_transition 수정** (316-325줄):
   ```python
   # 기존:
   # obs_extracted = self._extract_observation(single_obs)
   # obs_tensor = self._preprocess_observation(obs_extracted)

   # 변경:
   obs_tensor = self.preprocess_observation(single_obs)
   ```

### Step 4: 테스트

1. **유닛 테스트 실행** (있다면):
   ```bash
   python -m pytest tests/test_simple_agent.py -v
   ```

2. **실제 학습 테스트**:
   ```bash
   python main.py --mode train --config config/default.yaml
   ```

3. **검증 사항**:
   - 학습이 정상적으로 시작되는지
   - Loss가 계산되는지
   - Episode reward가 이전과 유사한지

---

## 예상 이점

### 1. 코드 중복 제거
- Simple Agent: ~50줄 감소
- 향후 DQN Agent: 전처리 코드 불필요 (Base 메서드만 사용)

### 2. 일관성 확보
- 모든 Agent가 동일한 전처리 로직 사용
- 버그 수정 시 한 곳만 수정하면 됨

### 3. 유지보수성 향상
- Observation space 처리 로직이 Base Agent에 집중
- 새로운 Agent 구현 시 네트워크 구조만 신경 쓰면 됨

---

## 주의사항

1. **Base Agent의 `preprocess_observation()` 시그니처 변경**
   - 기존 Base Agent를 사용하는 다른 코드가 있다면 영향 받을 수 있음
   - 현재는 Simple Agent만 있으므로 문제 없음

2. **Backward Compatibility**
   - RandomAgent는 전처리를 사용하지 않으므로 영향 없음
   - 향후 추가될 Agent들은 새로운 Base Agent 기준으로 작성

3. **테스트 필수**
   - 리팩터링 후 반드시 실제 학습 테스트 필요
   - Observation shape, 학습 진행 여부 확인

---

## 다음 작업자를 위한 체크리스트

- [ ] Base Agent `__init__`에 observation space 분석 코드 추가
- [ ] Base Agent `extract_observation()` 메서드 추가
- [ ] Base Agent `preprocess_observation()` 메서드 개선
- [ ] Simple Agent `__init__`에서 중복 코드 제거
- [ ] Simple Agent `_extract_observation`, `_preprocess_observation` 삭제
- [ ] Simple Agent `select_action`, `store_transition`에서 Base 메서드 사용
- [ ] `main.py --mode train` 실행하여 정상 동작 확인
- [ ] 이상 없으면 이 문서를 PROJECT.md에 반영

---

## 참고 파일 위치

- Base Agent: `agents/base_agent.py`
- Simple Agent: `agents/simple_agent.py`
- Main 실행: `main.py`
- 설정 파일: `config/default.yaml`

---

작성일: 2026-01-23
작성자: Claude (Sonnet 4.5)
