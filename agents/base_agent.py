"""
에이전트 베이스 클래스

이 모듈은 강화학습 에이전트의 표준 인터페이스를 정의합니다.
모든 구체적인 에이전트 구현은 이 베이스 클래스를 상속받아야 합니다.

중요: 이 프레임워크는 항상 VectorEnv를 가정합니다.
따라서 모든 입출력은 배치 형태입니다 (num_envs=1도 배치).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class BaseAgent(ABC):
    """
    강화학습 에이전트 베이스 클래스

    모든 에이전트는 이 인터페이스를 구현해야 합니다.
    환경과의 상호작용을 위한 표준 메서드를 정의합니다.

    중요: VectorEnv를 가정하므로 모든 관찰/행동/보상은 배치 형태입니다.
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
    def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False) -> np.ndarray:
        """
        관찰 배치를 받아 행동 배치를 선택

        Args:
            observation: 환경의 현재 상태 (배치)
                - Dict 형태: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array 형태: (N, ...)
            deterministic: True면 결정적 정책, False면 확률적 정책

        Returns:
            선택된 행동 배치 (N,) 형태의 ndarray
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        obs: Union[np.ndarray, Dict],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Union[np.ndarray, Dict],
        done: np.ndarray
    ) -> None:
        """
        Transition 배치를 저장 (학습과 분리)

        이 메서드는 transition을 내부 버퍼에 저장만 하고, 학습은 하지 않습니다.
        학습은 update() 메서드에서 별도로 수행됩니다.

        Args:
            obs: 현재 관찰 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            action: 선택한 행동 배치 (N,)
            reward: 받은 보상 배치 (N,)
            next_obs: 다음 관찰 배치 (obs와 동일한 형태)
            done: 에피소드 종료 여부 배치 (N,)

        Note:
            배치 크기 N은 VectorEnv의 num_envs와 동일합니다.
            num_envs=1이어도 배치 형태입니다 (N=1).
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        저장된 transition을 사용하여 에이전트 학습

        store_transition()으로 저장된 데이터를 바탕으로 학습을 수행합니다.
        언제 학습할지는 Trainer가 제어합니다.

        Returns:
            학습 메트릭 딕셔너리 (loss 등)
            학습이 일어나지 않았으면 빈 딕셔너리 {} 반환

        Note:
            - DQN 같은 off-policy: 매 N 스텝마다 호출
            - REINFORCE 같은 on-policy: 에피소드 완료 감지 후 호출
            - Random 에이전트: 학습하지 않음 (항상 {} 반환)
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

        # Observation space 분석
        from gymnasium import spaces as gym_spaces

        if isinstance(observation_space, gym_spaces.Dict):
            # Dict observation space
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

        # Observation shape 계산 (PyTorch 형식: C, H, W)
        if len(raw_obs_shape) == 3:
            # 이미지: (H, W, C) -> (C, H, W)
            num_channels = raw_obs_shape[2]
            self.obs_shape = (num_channels, raw_obs_shape[0], raw_obs_shape[1])
        else:
            # 벡터 입력
            self.obs_shape = raw_obs_shape

        # 신경망 모델 (하위 클래스에서 초기화)
        self.policy_net: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

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

    def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False) -> np.ndarray:
        """
        관찰 배치를 받아 행동 배치를 선택

        Args:
            observation: 환경의 현재 상태 배치
                - Dict: {'image': (N, H, W, C), 'score': (N, 1)}
                - Array: (N, ...)
            deterministic: 결정적 정책 사용 여부

        Returns:
            선택된 행동 배치 (N,)
        """
        if self.policy_net is None:
            # 정책 네트워크가 없으면 랜덤 행동
            if isinstance(observation, dict):
                batch_size = list(observation.values())[0].shape[0]
            else:
                batch_size = observation.shape[0]

            from gymnasium import spaces
            if isinstance(self.action_space, spaces.Discrete):
                return np.array([self.action_space.sample() for _ in range(batch_size)])
            else:
                return np.array([self.action_space.sample() for _ in range(batch_size)])

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
            'agent_type': self.__class__.__name__,  # 에이전트 타입 저장
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

    def select_action(self, observation: Union[np.ndarray, Dict], deterministic: bool = False) -> np.ndarray:
        """
        랜덤 행동 선택 (배치)

        Args:
            observation: 배치 관찰
            deterministic: 무시됨 (항상 랜덤)

        Returns:
            랜덤 행동 배치 (N,)
        """
        # 배치 크기 추출
        if isinstance(observation, dict):
            batch_size = list(observation.values())[0].shape[0]
        else:
            batch_size = observation.shape[0]

        # 배치 크기만큼 랜덤 행동 생성
        from gymnasium import spaces
        if isinstance(self.action_space, spaces.Discrete):
            return np.array([self.action_space.sample() for _ in range(batch_size)])
        else:
            return np.array([self.action_space.sample() for _ in range(batch_size)])

    def store_transition(
        self,
        obs: Union[np.ndarray, Dict],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Union[np.ndarray, Dict],
        done: np.ndarray
    ) -> None:
        """랜덤 에이전트는 transition을 저장하지 않음"""
        pass

    def update(self) -> Dict[str, float]:
        """랜덤 에이전트는 학습하지 않음"""
        return {}

    def save(self, path: str) -> None:
        """랜덤 에이전트는 저장할 것이 없음"""
        pass

    def load(self, path: str) -> None:
        """랜덤 에이전트는 로드할 것이 없음"""
        pass
