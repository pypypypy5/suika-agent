"""
Simple Policy Gradient Agent

가장 간단한 형태의 강화학습 에이전트입니다.
- Replay buffer 없음
- Target network 없음
- 단순 policy gradient로 학습
- 각 에피소드의 경험을 즉시 사용해 업데이트
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import RLAgent


class SimplePolicyNetwork(nn.Module):
    """
    간단한 정책 네트워크

    관찰을 입력받아 행동 확률을 출력합니다.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            obs_shape: 관찰 공간 shape
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

            # CNN 출력 크기 계산 (간단한 추정)
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
            self.encoder = nn.Identity()  # 벡터는 flatten 필요 없음
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
        Forward pass

        Args:
            x: 입력 관찰

        Returns:
            행동 로짓
        """
        # 벡터 입력은 (batch, features) 형태여야 함
        if not self.is_image and x.dim() == 1:
            x = x.unsqueeze(0)  # (features,) -> (1, features)

        features = self.encoder(x)
        logits = self.fc(features)
        return logits


class SimpleAgent(RLAgent):
    """
    Simple Policy Gradient Agent

    REINFORCE 알고리즘의 간단한 구현입니다.
    - 에피소드 단위로 학습
    - Monte Carlo return 사용
    - 즉시 업데이트 (no replay buffer)
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
            import numpy as np
            low = action_space.low[0]
            high = action_space.high[0]
            self.discrete_to_continuous = np.linspace(low, high, self.num_discrete_actions)
        else:
            raise NotImplementedError(
                f"SimpleAgent는 Discrete 또는 Box action space만 지원합니다. "
                f"받은 타입: {type(action_space)}"
            )

        # 관찰 공간 shape 확인 및 설정
        from gymnasium import spaces as gym_spaces

        if isinstance(observation_space, gym_spaces.Dict):
            # Dict observation space - 'image' 키 사용
            self.is_dict_obs = True
            self.obs_key = config.get('obs_key', 'image')

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
            # RGBA(4채널)는 RGB(3채널)로 변환할 것이므로 채널 수 조정
            num_channels = raw_obs_shape[2]
            if num_channels == 4:
                num_channels = 3  # RGBA -> RGB
            self.obs_shape = (num_channels, raw_obs_shape[0], raw_obs_shape[1])
        else:
            # 벡터 입력
            self.obs_shape = raw_obs_shape

        # 네트워크 설정
        hidden_dim = config.get('network', {}).get('hidden_dims', [128])[0]

        # 정책 네트워크 생성
        self.policy_net = SimplePolicyNetwork(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        # 에피소드 버퍼 (한 에피소드 동안의 경험 저장)
        self.reset_episode_buffer()

        # 통계
        self.episode_rewards = []

    def reset_episode_buffer(self):
        """에피소드 버퍼 초기화"""
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards_buffer: List[float] = []

    def _forward_policy(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        정책 네트워크를 통해 행동 선택

        Args:
            obs: 전처리된 관찰
            deterministic: True면 가장 높은 확률의 행동 선택

        Returns:
            선택된 행동
        """
        logits = self.policy_net(obs)

        if deterministic:
            # 결정적: argmax
            action = torch.argmax(logits, dim=-1)
        else:
            # 확률적: 샘플링
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            # 학습 모드일 때 log_prob 저장
            if self.policy_net.training:
                log_prob = dist.log_prob(action)
                self.episode_log_probs.append(log_prob)

        return action

    def _extract_observation(self, observation):
        """
        Dict observation에서 실제 관찰 추출

        Args:
            observation: 환경의 관찰 (Dict 또는 array)

        Returns:
            추출된 관찰 (array)
        """
        if self.is_dict_obs:
            # Dict에서 지정된 키 추출
            if isinstance(observation, dict):
                obs = observation[self.obs_key]
            else:
                # 이미 추출된 경우
                obs = observation
        else:
            obs = observation

        # RGBA -> RGB 변환 (4채널 -> 3채널)
        if isinstance(obs, np.ndarray) and len(obs.shape) == 3 and obs.shape[2] == 4:
            # Alpha 채널 제거
            obs = obs[:, :, :3]

        return obs

    def select_action(self, observation, deterministic: bool = False):
        """
        행동 선택

        Args:
            observation: 환경 관찰 (Dict 또는 array)
            deterministic: 결정적 선택 여부

        Returns:
            선택된 행동 (환경에 맞는 형태로 변환됨)
        """
        # Dict observation 처리
        obs = self._extract_observation(observation)

        with torch.no_grad():
            obs_tensor = self.preprocess_observation(obs)

            # 학습 모드에서는 gradient를 기록해야 함
            if self.policy_net.training and not deterministic:
                obs_tensor = self.preprocess_observation(obs)
                # gradient 필요
                obs_tensor.requires_grad = False
                with torch.enable_grad():
                    discrete_action = self._forward_policy(obs_tensor, deterministic)
            else:
                discrete_action = self._forward_policy(obs_tensor, deterministic)

        # 이산 행동을 numpy로 변환
        discrete_action_np = discrete_action.cpu().numpy()

        # scalar를 정수로 변환
        if discrete_action_np.shape == ():
            discrete_action_idx = int(discrete_action_np)
        else:
            discrete_action_idx = int(discrete_action_np[0])

        # 환경의 action space에 맞게 변환
        if self.is_discrete_env:
            # Discrete action space: 그대로 반환
            return discrete_action_idx
        else:
            # Box action space: 연속 값으로 변환
            continuous_value = self.discrete_to_continuous[discrete_action_idx]
            return np.array([continuous_value], dtype=np.float32)

    def store_transition(self, reward: float):
        """
        전이 저장 (보상만 저장, log_prob은 이미 저장됨)

        Args:
            reward: 받은 보상
        """
        self.episode_rewards_buffer.append(reward)
        # Note: total_steps는 Trainer에서 관리하므로 여기서 증가시키지 않음

    def update(
        self,
        obs: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        next_obs: Optional[np.ndarray] = None,
        done: bool = False
    ) -> Dict[str, float]:
        """
        에이전트 업데이트

        Trainer 호환성을 위해 매 스텝마다 호출되지만,
        실제 학습은 에피소드가 끝날 때만 수행합니다.

        Args:
            obs: 현재 관찰 (사용 안 함)
            action: 선택한 행동 (사용 안 함)
            reward: 받은 보상
            next_obs: 다음 관찰 (사용 안 함)
            done: 에피소드 종료 여부

        Returns:
            학습 메트릭 (에피소드 종료 시에만 반환)
        """
        # 보상 저장
        if reward is not None:
            self.store_transition(reward)

        # 에피소드가 끝나지 않았으면 빈 딕셔너리 반환
        if not done:
            return {}

        # 에피소드 종료 시 정책 업데이트
        if len(self.episode_log_probs) == 0:
            self.reset_episode_buffer()
            return {}

        # Monte Carlo returns 계산 (discounted cumulative rewards)
        returns = []
        G = 0
        for r in reversed(self.episode_rewards_buffer):
            G = r + self.gamma * G
            returns.insert(0, G)

        # 정규화 (학습 안정화)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss 계산
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)

        # 손실 합산 및 역전파
        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (학습 안정화)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 통계 저장
        episode_reward = sum(self.episode_rewards_buffer)
        self.episode_rewards.append(episode_reward)
        # Note: episodes는 Trainer에서 관리하므로 여기서 증가시키지 않음

        # 메트릭
        metrics = {
            'loss': loss.item(),
            'episode_reward': episode_reward,
            'episode_length': len(self.episode_rewards_buffer)
        }

        # 버퍼 초기화
        self.reset_episode_buffer()

        return metrics

    def get_statistics(self) -> Dict:
        """
        에이전트 통계 반환

        Returns:
            통계 딕셔너리
        """
        stats = super().get_statistics()

        if len(self.episode_rewards) > 0:
            stats.update({
                'mean_episode_reward': np.mean(self.episode_rewards[-100:]),
                'total_episodes': len(self.episode_rewards)
            })

        return stats
