"""
에이전트 베이스 클래스

이 모듈은 강화학습 에이전트의 표준 인터페이스를 정의합니다.
모든 구체적인 에이전트 구현은 이 베이스 클래스를 상속받아야 합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """
    강화학습 에이전트 베이스 클래스

    모든 에이전트는 이 인터페이스를 구현해야 합니다.
    환경과의 상호작용을 위한 표준 메서드를 정의합니다.
    """

    def __init__(self, observation_space: Any, action_space: Any, config: Optional[Dict] = None):
        """
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            config: 에이전트 설정 딕셔너리
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config or {}

        # 학습 통계
        self.total_steps = 0
        self.episodes = 0

    @abstractmethod
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        관찰을 받아 행동을 선택

        Args:
            observation: 환경의 현재 상태
            deterministic: True면 결정적 정책, False면 확률적 정책

        Returns:
            선택된 행동
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        에이전트 학습 업데이트

        Returns:
            학습 메트릭 딕셔너리 (loss, etc.)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        에이전트 모델 저장

        Args:
            path: 저장 경로
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        에이전트 모델 로드

        Args:
            path: 로드 경로
        """
        pass

    def train(self) -> None:
        """학습 모드로 전환"""
        pass

    def eval(self) -> None:
        """평가 모드로 전환"""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """
        에이전트 통계 반환

        Returns:
            통계 딕셔너리
        """
        return {
            'total_steps': self.total_steps,
            'episodes': self.episodes
        }


class RLAgent(BaseAgent):
    """
    딥러닝 기반 강화학습 에이전트

    PyTorch를 사용하는 에이전트의 베이스 클래스
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        config: Optional[Dict] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            config: 에이전트 설정
            device: PyTorch 디바이스 ('cuda', 'cpu', 또는 None for auto)
        """
        super().__init__(observation_space, action_space, config)

        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 학습 파라미터
        self.gamma = config.get('gamma', 0.99)  # 할인율
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.batch_size = config.get('batch_size', 64)

        # 신경망 모델 (하위 클래스에서 초기화)
        self.policy_net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def preprocess_observation(self, obs: np.ndarray) -> torch.Tensor:
        """
        관찰을 신경망 입력으로 전처리

        Args:
            obs: 원본 관찰

        Returns:
            전처리된 텐서
        """
        if isinstance(obs, dict):
            # 딕셔너리 형태의 관찰 처리
            # 예: {'image': ..., 'score': ...}
            obs = obs.get('image', list(obs.values())[0])

        # NumPy 배열을 텐서로 변환
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)

        # 배치 차원이 없으면 추가
        if obs.dim() == 3:  # (H, W, C)
            obs = obs.unsqueeze(0)  # (1, H, W, C)

        # 이미지면 채널을 앞으로 (PyTorch 컨벤션)
        if obs.dim() == 4 and obs.shape[-1] in [1, 3, 4]:  # (B, H, W, C)
            obs = obs.permute(0, 3, 1, 2)  # (B, C, H, W)

        return obs.to(self.device)

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        관찰을 받아 행동을 선택

        Args:
            observation: 환경의 현재 상태
            deterministic: 결정적 정책 사용 여부

        Returns:
            선택된 행동
        """
        if self.policy_net is None:
            # 정책 네트워크가 없으면 랜덤 행동
            return self.action_space.sample()

        # 신경망으로 행동 선택
        with torch.no_grad():
            obs_tensor = self.preprocess_observation(observation)
            action = self._forward_policy(obs_tensor, deterministic)

        return action.cpu().numpy()

    @abstractmethod
    def _forward_policy(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """
        정책 네트워크 forward pass

        Args:
            obs: 전처리된 관찰 텐서
            deterministic: 결정적 정책 사용 여부

        Returns:
            행동 텐서
        """
        pass

    def train(self) -> None:
        """학습 모드로 전환"""
        if self.policy_net is not None:
            self.policy_net.train()

    def eval(self) -> None:
        """평가 모드로 전환"""
        if self.policy_net is not None:
            self.policy_net.eval()

    def save(self, path: str) -> None:
        """
        에이전트 모델 저장

        Args:
            path: 저장 경로
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict() if self.policy_net else None,
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'total_steps': self.total_steps,
            'episodes': self.episodes,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """
        에이전트 모델 로드

        Args:
            path: 로드 경로
        """
        checkpoint = torch.load(path, map_location=self.device)

        if self.policy_net and checkpoint.get('policy_net'):
            self.policy_net.load_state_dict(checkpoint['policy_net'])

        if self.optimizer and checkpoint.get('optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.total_steps = checkpoint.get('total_steps', 0)
        self.episodes = checkpoint.get('episodes', 0)

        print(f"Model loaded from {path}")


class RandomAgent(BaseAgent):
    """
    랜덤 행동을 선택하는 베이스라인 에이전트

    성능 비교를 위한 간단한 에이전트
    """

    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """랜덤 행동 선택"""
        return self.action_space.sample()

    def update(self, *args, **kwargs) -> Dict[str, float]:
        """랜덤 에이전트는 학습하지 않음"""
        return {}

    def save(self, path: str) -> None:
        """랜덤 에이전트는 저장할 것이 없음"""
        pass

    def load(self, path: str) -> None:
        """랜덤 에이전트는 로드할 것이 없음"""
        pass
