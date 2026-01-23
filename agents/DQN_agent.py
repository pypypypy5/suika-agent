from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import RLAgent


class ReplayBuffer:
      def __init__(self, capacity):
          self.buffer = deque(maxlen=capacity)

      def push(self, state, action, reward, next_state, done):
          self.buffer.append((state, action, reward, next_state, done))

      def sample(self, batch_size):
          batch = random.sample(self.buffer, batch_size)
          states, actions, rewards, next_states, dones = zip(*batch)
          return (
              np.array(states),
              np.array(actions),
              np.array(rewards),
              np.array(next_states),
              np.array(dones)
          )

      def __len__(self):
          return len(self.buffer)

class DQNNetwork(nn.Module):
    """
    DQN Q-Network

    관찰 배치를 입력받아 각 행동의 Q-값 배치를 출력합니다.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            obs_shape: 관찰 공간 shape (C, H, W) 또는 (features,)
            action_dim: 행동 공간 차원
            hidden_dim: 은닉층 크기
        """
        super().__init__()

        # 입력 크기 계산
        if len(obs_shape) == 3:  # 이미지 입력 (C, H, W)
            # 간단한 CNN
            self.encoder = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )

            # CNN 출력 크기 계산
            with torch.no_grad():
                sample_input = torch.zeros(1, *obs_shape)
                cnn_output_size = self.encoder(sample_input).shape[1]

            self.fc = nn.Sequential(
                nn.Linear(cnn_output_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.is_image = True
        else:
            # 벡터 입력
            input_dim = np.prod(obs_shape)
            self.encoder = nn.Identity()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.is_image = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (배치 지원)

        Args:
            x: 입력 관찰 (N, C, H, W) 또는 (N, features)

        Returns:
            Q-values (N, action_dim) - 각 행동의 가치 추정
        """
        features = self.encoder(x)
        q_values = self.fc(features)
        return q_values


class DQNAgent(RLAgent):
    """
    DQN (Deep Q-Network) Agent (VectorEnv 지원)

    DQN 알고리즘의 구현입니다.
    - Experience Replay Buffer 사용
    - Target Network로 학습 안정화
    - Epsilon-greedy exploration
    - TD Learning (Temporal Difference)
    - VectorEnv의 각 환경에서 transition 수집
    """

    def __init__(
        self,
        observation_space,
        action_space,
        config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            config: 설정 딕셔너리
            device: PyTorch 디바이스
        """
        # 설정 기본값
        if config is None:
            config = {}

        super().__init__(observation_space, action_space, config, device)

        # 행동 공간 확인 및 설정
        if hasattr(action_space, 'n'):
            # Discrete action space
            self.action_dim = action_space.n
            self.is_discrete_env = True
            self.continuous_action_space = None
        elif hasattr(action_space, 'shape'):
            # Box (continuous) action space
            # 내부적으로 이산화하여 처리
            self.num_discrete_actions = config.get('num_discrete_actions', 11)
            self.action_dim = self.num_discrete_actions
            self.is_discrete_env = False
            self.continuous_action_space = action_space

            # 이산 행동 -> 연속 행동 매핑 테이블
            low = action_space.low[0]
            high = action_space.high[0]
            self.discrete_to_continuous = np.linspace(low, high, self.num_discrete_actions)
        else:
            raise NotImplementedError(
                f"SimpleAgent는 Discrete 또는 Box action space만 지원합니다. "
                f"받은 타입: {type(action_space)}"
            )

        # 네트워크 설정
        hidden_dim = config.get('network', {}).get('hidden_dims', [128])[0] if 'network' in config else 128

        # Q-Networks (Policy + Target)
        self.policy_net = DQNNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_net = DQNNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # Target network를 policy network로 초기화
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network는 항상 eval 모드

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        # Epsilon-greedy 파라미터
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)

        # Target network 업데이트 빈도
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.update_counter = 0

        # Replay Buffer
        buffer_capacity = config.get('buffer_capacity', 100000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # 통계
        self.episode_rewards = []

    def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False) -> np.ndarray:
        """
        관찰 배치를 받아 행동 배치 선택 (Epsilon-greedy)

        Args:
            observation: 환경의 관찰 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            deterministic: True면 greedy (argmax), False면 epsilon-greedy

        Returns:
            선택된 행동 배치 (N,)
        """
        # 배치 크기 확인
        if isinstance(observation, dict):
            batch_size = list(observation.values())[0].shape[0]
        else:
            batch_size = observation.shape[0]

        # Epsilon-greedy
        if not deterministic and np.random.random() < self.epsilon:
            # exploration
            discrete_actions_np = np.random.randint(0, self.action_dim, size=batch_size)
        else:
            # exploitation
            obs_tensor = self.preprocess_observation(observation)  # (N, C, H, W)

            with torch.no_grad():
                q_values = self.policy_net(obs_tensor)  # (N, action_dim)
                discrete_actions = q_values.argmax(dim=1)  # (N,) - 최대 Q값 행동 선택

            discrete_actions_np = discrete_actions.cpu().numpy()  # (N,)

        # 환경의 action space에 맞게 변환
        if self.is_discrete_env:
            # Discrete action space: 그대로 반환
            return discrete_actions_np
        else:
            # Box action space: 이산 -> 연속 변환
            continuous_values = self.discrete_to_continuous[discrete_actions_np]
            return continuous_values.reshape(-1, 1).astype(np.float32)

    def store_transition(
        self,
        obs: Union[np.ndarray, Dict],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Union[np.ndarray, Dict],
        done: np.ndarray
    ) -> None:
        """
        Transition 배치를 Replay Buffer에 저장 (DQN)

        Args:
            obs: 현재 관찰 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            action: 선택한 행동 배치 (N,)
            reward: 받은 보상 배치 (N,)
            next_obs: 다음 관찰 배치
            done: 에피소드 종료 여부 배치 (N,)
        """
        # 학습 모드일 때만 저장
        if not self.policy_net.training:
            return

        batch_size = len(done)

        # 관찰 추출 (Dict -> Array)
        obs_array = self.extract_observation(obs)  # (N, H, W, C)
        next_obs_array = self.extract_observation(next_obs)  # (N, H, W, C)

        # 각 환경마다 replay buffer에 저장
        for env_id in range(batch_size):
            # 행동 값 변환
            if self.is_discrete_env:
                action_value = int(action[env_id])
            else:
                # Box action space: 연속 값을 이산 인덱스로 역변환
                continuous_value = action[env_id][0] if len(action[env_id].shape) > 0 else action[env_id]
                action_value = int(np.argmin(np.abs(self.discrete_to_continuous - continuous_value)))

            # Replay buffer에 저장
            self.replay_buffer.push(
                state=obs_array[env_id],          # (H, W, C)
                action=action_value,               # int
                reward=float(reward[env_id]),      # float
                next_state=next_obs_array[env_id], # (H, W, C)
                done=float(done[env_id])           # float (0 or 1)
            )

    def update(self) -> Dict[str, float]:
        """
        Replay Buffer에서 샘플링하여 DQN 학습 (TD Learning)

        Returns:
            {'loss': float, 'epsilon': float} 또는 {}
        """
        # Replay buffer에 충분한 데이터가 없으면 학습 스킵
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 1. Replay buffer에서 랜덤 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 2. NumPy to Tensor 변환
        states = torch.FloatTensor(states).to(self.device)  # (batch, H, W, C)
        actions = torch.LongTensor(actions).to(self.device)  # (batch,)
        rewards = torch.FloatTensor(rewards).to(self.device)  # (batch,)
        next_states = torch.FloatTensor(next_states).to(self.device)  # (batch, H, W, C)
        dones = torch.FloatTensor(dones).to(self.device)  # (batch,)

        # 3. 이미지 전처리: (batch, H, W, C) -> (batch, C, H, W)
        if len(states.shape) == 4:
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)

        # 4. Current Q-values: Q(s, a)
        current_q_values = self.policy_net(states)  # (batch, action_dim)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch,)

        # 5. Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # (batch, action_dim)
            next_q = next_q_values.max(1)[0]  # (batch,) - 최대 Q값
            target_q = rewards + self.gamma * next_q * (1 - dones)  # (batch,)

        # 6. Loss 계산 (Huber Loss = Smooth L1)
        loss = F.smooth_l1_loss(current_q, target_q)

        # 7. 역전파
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # 8. Epsilon 감소 (exploration → exploitation)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 9. Target network 주기적 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_mean': current_q.mean().item()
        }

    def _forward_policy(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        정책 네트워크 forward pass (DQN은 select_action에서 직접 처리)

        Args:
            obs: 전처리된 관찰 텐서
            deterministic: 결정적 정책 사용 여부

        Returns:
            행동 텐서
        """
        # DQN은 epsilon-greedy를 select_action에서 처리하므로
        # 여기서는 greedy action만 반환
        q_values = self.policy_net(obs)
        return q_values.argmax(dim=1)

    def get_statistics(self) -> Dict:
        """
        에이전트 통계 반환

        Returns:
            통계 딕셔너리
        """
        stats = super().get_statistics()

        # DQN 특화 통계
        stats['epsilon'] = self.epsilon
        stats['buffer_size'] = len(self.replay_buffer)

        return stats
