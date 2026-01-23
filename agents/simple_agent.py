"""
Simple Policy Gradient Agent (VectorEnv 지원)

가장 간단한 형태의 강화학습 에이전트입니다.
- Replay buffer 없음
- Target network 없음
- 단순 policy gradient로 학습
- 각 에피소드의 경험을 즉시 사용해 업데이트
- VectorEnv 배치 처리 지원
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_agent import RLAgent


class SimplePolicyNetwork(nn.Module):
    """
    간단한 정책 네트워크

    관찰 배치를 입력받아 행동 확률 배치를 출력합니다.
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
            행동 로짓 (N, action_dim)
        """
        features = self.encoder(x)
        logits = self.fc(features)
        return logits


class SimpleAgent(RLAgent):
    """
    Simple Policy Gradient Agent (VectorEnv 지원)

    REINFORCE 알고리즘의 간단한 구현입니다.
    - 에피소드 단위로 학습
    - Monte Carlo return 사용
    - VectorEnv의 각 환경별로 독립적인 버퍼 유지
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

        # 환경별 에피소드 버퍼
        self.episode_buffers: Dict[int, Dict[str, List]] = {}  # {env_id: {'observations': [], 'actions': [], 'rewards': []}}
        self.completed_episodes: set = set()  # 학습 준비된 에피소드들

        # 통계
        self.episode_rewards = []

    def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False) -> np.ndarray:
        """
        관찰 배치를 받아 행동 배치 선택

        Args:
            observation: 환경의 관찰 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            deterministic: True면 결정적 행동, False면 확률적 행동

        Returns:
            선택된 행동 배치 (N,)
        """
        # Base Agent의 전처리 메서드 사용
        obs_tensor = self.preprocess_observation(observation)  # (N, C, H, W)

        # 네트워크 forward
        with torch.no_grad():
            logits = self.policy_net(obs_tensor)  # (N, action_dim)
            probs = F.softmax(logits, dim=-1)  # (N, action_dim)

            if deterministic:
                # 결정적: argmax
                discrete_actions = probs.argmax(dim=1)  # (N,)
            else:
                # 확률적: 샘플링
                dist = Categorical(probs=probs)
                discrete_actions = dist.sample()  # (N,)

        # numpy로 변환
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
        Transition 배치를 환경별로 저장

        Args:
            obs: 현재 관찰 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            action: 선택한 행동 배치 (N,)
            reward: 받은 보상 배치 (N,)
            next_obs: 다음 관찰 배치
            done: 에피소드 종료 여부 배치 (N,)
        """
        batch_size = len(done)

        # 각 환경마다 처리
        for env_id in range(batch_size):
            # 버퍼 초기화
            if env_id not in self.episode_buffers:
                self.episode_buffers[env_id] = {
                    'observations': [],
                    'actions': [],
                    'rewards': []
                }

            # 학습 모드일 때만 저장
            if self.policy_net.training:
                # 단일 관찰 저장
                if isinstance(obs, dict):
                    single_obs = {k: v[env_id:env_id+1] for k, v in obs.items()}
                else:
                    single_obs = obs[env_id:env_id+1]

                # 행동 값 저장
                if self.is_discrete_env:
                    action_value = int(action[env_id])
                else:
                    # Box action space: 연속 값을 이산 인덱스로 역변환
                    continuous_value = action[env_id][0] if len(action[env_id].shape) > 0 else action[env_id]
                    action_value = int(np.argmin(np.abs(self.discrete_to_continuous - continuous_value)))

                # 저장
                self.episode_buffers[env_id]['observations'].append(single_obs)
                self.episode_buffers[env_id]['actions'].append(action_value)
                self.episode_buffers[env_id]['rewards'].append(float(reward[env_id]))

            # 에피소드 종료 시
            if done[env_id]:
                if self.policy_net.training and len(self.episode_buffers[env_id]['rewards']) > 0:
                    self.completed_episodes.add(env_id)
                else:
                    # 평가 모드거나 빈 버퍼면 초기화
                    self.episode_buffers[env_id] = {'observations': [], 'actions': [], 'rewards': []}

    def update(self) -> Dict[str, float]:
        """
        완료된 에피소드들에 대해 REINFORCE 학습

        Returns:
            {'loss': float, 'num_episodes_updated': int}
        """
        if not self.completed_episodes:
            return {}

        total_loss = 0.0
        num_episodes = 0
        total_reward = 0.0

        # 모든 에피소드의 loss를 모아서 한번에 최적화
        all_log_probs = []
        all_returns = []

        for env_id in list(self.completed_episodes):
            buffer = self.episode_buffers[env_id]

            if len(buffer['rewards']) == 0:
                continue

            # Monte Carlo returns 계산
            returns = []
            G = 0
            for r in reversed(buffer['rewards']):
                G = r + self.gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, device=self.device)

            # 정규화
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Log probs 재계산 (gradient를 위해 필요)
            log_probs = []
            for obs, act in zip(buffer['observations'], buffer['actions']):
                # 전처리
                obs_tensor = self.preprocess_observation(obs)  # (1, C, H, W)

                # Log prob 계산
                logits = self.policy_net(obs_tensor)  # (1, action_dim)
                probs = F.softmax(logits, dim=-1)  # (1, action_dim)
                dist = Categorical(probs=probs)

                action_tensor = torch.tensor([act], device=self.device)
                log_prob = dist.log_prob(action_tensor)  # (1,)
                log_probs.append(log_prob)

            log_probs = torch.cat(log_probs)  # (T,)

            all_log_probs.append(log_probs)
            all_returns.append(returns)

            num_episodes += 1
            total_reward += sum(buffer['rewards'])

            # 통계 저장
            self.episode_rewards.append(sum(buffer['rewards']))

            # 버퍼 초기화
            self.episode_buffers[env_id] = {'observations': [], 'actions': [], 'rewards': []}

        # 최적화 (모든 완료된 에피소드에 대해 한번에)
        if num_episodes > 0:
            # 모든 log_probs와 returns 합치기
            combined_log_probs = torch.cat(all_log_probs)
            combined_returns = torch.cat(all_returns)

            # Policy gradient loss
            loss = -(combined_log_probs * combined_returns).mean()

            # 역전파
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss = loss.item()

        # 학습 완료된 에피소드 제거
        self.completed_episodes.clear()

        if num_episodes == 0:
            return {}

        return {
            'loss': total_loss,
            'num_episodes_updated': num_episodes,
            'mean_episode_reward': total_reward / num_episodes
        }

    def _forward_policy(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        정책 네트워크를 통해 행동 배치 선택

        Args:
            obs: 전처리된 관찰 배치 (N, C, H, W)
            deterministic: True면 가장 높은 확률의 행동 선택

        Returns:
            선택된 행동 배치 (N,)
        """
        logits = self.policy_net(obs)  # (N, action_dim)

        if deterministic:
            # 결정적: argmax
            action = torch.argmax(logits, dim=-1)  # (N,)
        else:
            # 확률적: 샘플링
            probs = F.softmax(logits, dim=-1)  # (N, action_dim)
            dist = Categorical(probs=probs)
            action = dist.sample()  # (N,)

        return action

    def get_statistics(self) -> Dict:
        """
        에이전트 통계 반환

        Returns:
            통계 딕셔너리
        """
        stats = super().get_statistics()

        if len(self.episode_rewards) > 0:
            stats['mean_episode_reward'] = np.mean(self.episode_rewards[-100:])
            stats['max_episode_reward'] = np.max(self.episode_rewards)
            stats['min_episode_reward'] = np.min(self.episode_rewards)

        return stats
