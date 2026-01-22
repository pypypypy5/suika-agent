# VectorEnv 병렬 환경 리팩터링 가이드

## 완료된 작업

### 1. BaseAgent 인터페이스 수정 ✅
- `store_transition()` 추상 메서드 추가
- 모든 메서드를 배치 처리용으로 문서화
- `select_action()`: (N, ...) 입력 → (N,) 출력
- `store_transition()`: 배치 transition 저장
- `update()`: 저장된 데이터로 학습

파일: `agents/base_agent.py`

### 2. create_env() 수정 ✅
- 항상 VectorEnv 반환 (num_envs=1도 VectorEnv)
- num_envs=1: SyncVectorEnv (오버헤드 최소)
- num_envs>1: AsyncVectorEnv (병렬 처리)
- 각 환경마다 고유 포트 할당 (port + rank)

파일: `main.py`

### 3. 통합 테스트 작성 ✅
- VectorEnv 생성 테스트
- 에이전트 배치 처리 테스트
- Trainer 통합 테스트
- 성능 테스트

파일: `tests/test_unified_vector_env.py`

---

## 미완료 작업

### 1. SimpleAgent 배치 처리 지원 (진행중)

#### 필요한 변경사항:

**a) 환경별 버퍼 관리**
```python
class SimpleAgent(RLAgent):
    def __init__(self, ...):
        # 환경별 에피소드 버퍼
        self.episode_buffers = {}  # {env_id: {'log_probs': [], 'rewards': []}}
        self.completed_episodes = set()  # 학습 준비된 env_id들
```

**b) select_action() - 배치 처리**
```python
def select_action(self, observation: Dict, deterministic: bool = False) -> np.ndarray:
    """
    배치 관찰 → 배치 행동

    Input: {'image': (N, H, W, C), 'score': (N, 1)}
    Output: (N,) actions
    """
    # 1. 관찰 추출
    obs = self._extract_observation(observation)  # (N, H, W, C)

    # 2. 전처리
    obs_tensor = self._preprocess_observation(obs)  # (N, C, H, W)

    # 3. 네트워크 forward
    with torch.no_grad():
        logits = self.policy_net(obs_tensor)  # (N, action_dim)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            actions = probs.argmax(dim=1)  # (N,)
        else:
            dist = Categorical(probs=probs)
            actions = dist.sample()  # (N,)

    # 4. numpy 변환
    actions_np = actions.cpu().numpy()  # (N,)

    # 5. 환경 타입에 맞게 변환
    if self.is_discrete_env:
        return actions_np
    else:
        # Box action space: 이산 → 연속
        return self.discrete_to_continuous[actions_np].reshape(-1, 1)
```

**c) store_transition() - 환경별 저장**
```python
def store_transition(
    self,
    obs: Dict,
    action: np.ndarray,
    reward: np.ndarray,
    next_obs: Dict,
    done: np.ndarray
) -> None:
    """
    배치 transition을 환경별로 분리하여 저장

    Args:
        obs: {'image': (N, H, W, C), ...}
        action: (N,)
        reward: (N,)
        next_obs: {...}
        done: (N,)
    """
    batch_size = len(done)

    for env_id in range(batch_size):
        # 버퍼 초기화
        if env_id not in self.episode_buffers:
            self.episode_buffers[env_id] = {
                'log_probs': [],
                'rewards': []
            }

        # 학습 모드일 때만 log_prob 계산
        if self.policy_net.training:
            # 단일 관찰 추출
            single_obs = {k: v[env_id:env_id+1] for k, v in obs.items()}

            # Log prob 계산
            obs_extracted = self._extract_observation(single_obs)
            obs_tensor = self._preprocess_observation(obs_extracted)  # (1, C, H, W)

            logits = self.policy_net(obs_tensor)  # (1, action_dim)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)

            action_tensor = torch.tensor([action[env_id]], device=self.device)
            log_prob = dist.log_prob(action_tensor)

            # 저장
            self.episode_buffers[env_id]['log_probs'].append(log_prob)
            self.episode_buffers[env_id]['rewards'].append(float(reward[env_id]))

        # 에피소드 종료 시
        if done[env_id]:
            if self.policy_net.training and len(self.episode_buffers[env_id]['rewards']) > 0:
                self.completed_episodes.add(env_id)
            else:
                # 평가 모드거나 빈 버퍼면 초기화
                self.episode_buffers[env_id] = {'log_probs': [], 'rewards': []}
```

**d) update() - 완료된 에피소드들 학습**
```python
def update(self) -> Dict[str, float]:
    """
    완료된 에피소드들에 대해 REINFORCE 학습
    """
    if not self.completed_episodes:
        return {}

    total_loss = 0.0
    num_episodes = 0

    # 각 완료된 에피소드에 대해
    for env_id in list(self.completed_episodes):
        buffer = self.episode_buffers[env_id]

        if len(buffer['rewards']) == 0:
            continue

        # 1. Returns 계산
        returns = []
        G = 0
        for r in reversed(buffer['rewards']):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device)

        # 2. 정규화
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 3. Loss 계산
        log_probs = torch.cat(buffer['log_probs'])  # (T,)
        loss = -(log_probs * returns).sum()

        # 4. 최적화
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        total_loss += loss.item()
        num_episodes += 1

        # 통계 저장
        self.episode_rewards.append(sum(buffer['rewards']))

        # 버퍼 초기화
        self.episode_buffers[env_id] = {'log_probs': [], 'rewards': []}

    # 완료된 에피소드 제거
    self.completed_episodes.clear()

    if num_episodes == 0:
        return {}

    return {
        'loss': total_loss / num_episodes,
        'num_episodes_updated': num_episodes
    }
```

#### 수정 방법:
1. `agents/simple_agent.py` 백업 완료 (`simple_agent.py.backup`)
2. 위 코드 참고하여 메서드들 수정
3. `_forward_policy()` 수정: 배치 입력 지원
4. `_extract_observation()`, `_preprocess_observation()` 수정: 배치 지원

---

### 2. Trainer 수정

파일: `training/trainer.py`

#### 필요한 변경사항:

**a) 학습 루프 단순화**
```python
class Trainer:
    def train(self):
        """VectorEnv 전용 학습 루프"""
        self.agent.train()

        obs, info = self.env.reset()
        num_envs = self.env.num_envs

        # 환경별 통계
        episode_rewards = [0.0] * num_envs
        episode_lengths = [0] * num_envs

        for step in range(self.total_timesteps):
            # 1. 행동 선택 (배치)
            actions = self.agent.select_action(obs, deterministic=False)

            # 2. 환경 스텝 (배치)
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            dones = terminated | truncated

            # 3. Transition 저장 (배치)
            self.agent.store_transition(obs, actions, rewards, next_obs, dones)

            # 4. 에피소드 통계 업데이트
            for env_id in range(num_envs):
                episode_rewards[env_id] += rewards[env_id]
                episode_lengths[env_id] += 1

                if dones[env_id]:
                    # 로깅
                    if self.logger:
                        self.logger.log_scalar(
                            'train/episode_reward',
                            episode_rewards[env_id],
                            step
                        )

                    episode_rewards[env_id] = 0.0
                    episode_lengths[env_id] = 0

            # 5. 학습 (주기적으로)
            if step % self.config.training.update_frequency == 0:
                update_info = self.agent.update()

                if update_info and self.logger:
                    for key, value in update_info.items():
                        self.logger.log_scalar(f'train/{key}', value, step)

            obs = next_obs

        self.env.close()
```

**핵심 변경점**:
- `is_vector_env` 분기 제거 (항상 VectorEnv)
- `store_transition()` + `update()` 분리
- 환경별 통계를 배열로 관리

---

### 3. 기존 테스트 업데이트

#### 수정 필요한 테스트들:

**a) `tests/test_simple_agent.py`**
- Mock 환경을 VectorEnv로 감싸기
- 모든 관찰/행동을 배치 형태로 수정
- `store_transition()` 호출 방식 변경

예시:
```python
def test_agent_select_action_batch():
    """배치 행동 선택"""
    obs_batch = {
        'image': np.random.randint(0, 256, (4, 84, 84, 4), dtype=np.uint8),
        'score': np.random.rand(4, 1).astype(np.float32)
    }

    actions = agent.select_action(obs_batch)
    assert actions.shape == (4,)
```

**b) `tests/test_agent_trainer_integration.py`**
- MockEnv를 VectorEnv로 감싸기
- `simulate_trainer_loop()` 수정

---

## 테스트 실행 방법

### 1. 새 테스트 실행
```bash
pytest tests/test_unified_vector_env.py -v
```

### 2. 전체 테스트 실행
```bash
pytest tests/ -v
```

### 3. 특정 테스트만
```bash
pytest tests/test_simple_agent.py::TestSimpleAgent::test_agent_select_action_batch -v
```

---

## 체크리스트

- [x] BaseAgent 인터페이스 수정
- [x] RandomAgent 배치 처리 지원
- [x] create_env() VectorEnv 반환
- [x] 통합 테스트 작성
- [ ] SimpleAgent 완전 수정
- [ ] Trainer 수정
- [ ] 기존 테스트 업데이트
- [ ] 전체 테스트 통과 확인

---

## 다음 단계

1. **SimpleAgent 수정 완료**
   - 위 가이드 참고
   - `agents/simple_agent.py` 수정

2. **Trainer 수정**
   - `training/trainer.py` 수정

3. **테스트 업데이트**
   - 기존 테스트들을 배치 형태로 수정

4. **통합 테스트**
   - `pytest tests/ -v` 실행
   - 모든 테스트 통과 확인

5. **성능 벤치마크**
   - 단일 환경 vs 4개 환경 throughput 비교
   - `tests/test_unified_vector_env.py::TestPerformanceWithVectorEnv` 실행

---

## 참고사항

### VectorEnv의 auto-reset
VectorEnv는 `done=True`인 환경을 자동으로 reset합니다.
- `next_obs`에 새 에피소드의 첫 관찰 포함
- `info`에 `final_observation`, `final_info` 포함

### 배치 크기
- 단일 환경: batch_size=1, shape=(1, ...)
- 4개 환경: batch_size=4, shape=(4, ...)

### 디버깅 팁
```python
# 배치 크기 확인
print(f"obs shape: {obs['image'].shape}")  # (N, H, W, C)
print(f"actions shape: {actions.shape}")   # (N,)
print(f"rewards shape: {rewards.shape}")   # (N,)
```
