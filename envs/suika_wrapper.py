"""
Suika Game 환경 래퍼

이 모듈은 suika_rl 환경을 감싸서 에이전트가 환경의 세부 구현을
알 필요 없이 표준 인터페이스로 상호작용할 수 있도록 합니다.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional

# suika_rl 패키지 경로 추가
suika_rl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'suika_rl')
if os.path.exists(suika_rl_path) and suika_rl_path not in sys.path:
    sys.path.insert(0, suika_rl_path)


class SuikaEnvWrapper(gym.Wrapper):
    """
    Suika Game 환경의 래퍼 클래스

    이 클래스는 suika_rl 환경을 감싸서 다음을 제공합니다:
    - 표준화된 observation/action 인터페이스
    - 보상 함수 커스터마이징
    - 추가적인 상태 정보 추출
    - 환경 세부사항 캡슐화

    Attributes:
        env: 원본 Suika 환경
        observation_type: 관찰 타입 ('image', 'features', 'both')
        reward_scale: 보상 스케일링 팩터
    """

    def __init__(
        self,
        headless: bool = True,
        port: int = 8923,
        delay_before_img_capture: float = 0.5,
        observation_type: str = "image",
        reward_scale: float = 1.0,
        normalize_obs: bool = True,
        use_mock: bool = False,
        **kwargs
    ):
        """
        Args:
            headless: Selenium 브라우저 headless 모드 사용 여부
            port: 로컬 HTTP 서버 포트
            delay_before_img_capture: 이미지 캡처 전 대기 시간 (초)
            observation_type: 관찰 데이터 타입
                - 'image': 게임 화면 이미지
                - 'features': 추출된 특징 벡터
                - 'both': 이미지와 특징 모두
            reward_scale: 보상 값 스케일링 팩터
            normalize_obs: 관찰 정규화 여부
            use_mock: True면 실제 환경 대신 Mock 환경 사용 (개발/테스트용)
            **kwargs: 환경 생성에 전달할 추가 인자
        """
        # Mock 환경 또는 실제 환경 선택
        if use_mock:
            print("Using mock environment for development/testing.")
            base_env = self._create_mock_env()
        else:
            try:
                # suika_rl에서 SuikaBrowserEnv import
                from suika_env.suika_browser_env import SuikaBrowserEnv
                base_env = SuikaBrowserEnv(
                    headless=headless,
                    port=port,
                    delay_before_img_capture=delay_before_img_capture
                )
                print(f"Using real Suika environment (headless={headless}, port={port})")
            except ImportError as e:
                print(f"Warning: Could not import SuikaBrowserEnv: {e}")
                print("Falling back to mock environment. Install suika_rl to use real environment.")
                base_env = self._create_mock_env()
            except Exception as e:
                print(f"Warning: Error creating Suika environment: {e}")
                print("Falling back to mock environment.")
                base_env = self._create_mock_env()

        super().__init__(base_env)

        self.observation_type = observation_type
        self.reward_scale = reward_scale
        self.normalize_obs = normalize_obs

        # 에피소드 통계
        self.episode_score = 0
        self.episode_steps = 0
        self.best_score = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        환경 리셋

        Args:
            seed: 랜덤 시드
            options: 리셋 옵션

        Returns:
            observation: 초기 관찰
            info: 추가 정보 딕셔너리
        """
        obs, info = self.env.reset(seed=seed, options=options)

        # 에피소드 통계 초기화
        self.episode_score = 0
        self.episode_steps = 0

        # 관찰 전처리
        processed_obs = self._process_observation(obs)

        # 추가 정보
        info.update({
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'best_score': self.best_score
        })

        return processed_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 실행

        Args:
            action: 에이전트의 행동 (과일을 떨어뜨릴 위치)

        Returns:
            observation: 다음 상태 관찰
            reward: 받은 보상
            terminated: 에피소드 종료 여부
            truncated: 시간 제한 등으로 인한 조기 종료
            info: 추가 정보
        """
        # 행동 실행
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 에피소드 통계 업데이트
        self.episode_steps += 1
        self.episode_score += reward

        # 보상 처리
        processed_reward = self._process_reward(reward, info)

        # 관찰 전처리
        processed_obs = self._process_observation(obs)

        # 추가 정보 업데이트
        info.update({
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'original_reward': reward,
            'processed_reward': processed_reward
        })

        # 에피소드 종료 시 최고 점수 업데이트
        if terminated or truncated:
            if self.episode_score > self.best_score:
                self.best_score = self.episode_score
            info['best_score'] = self.best_score

        return processed_obs, processed_reward, terminated, truncated, info

    def _process_observation(self, obs: Any) -> Dict[str, Any]:
        """
        관찰 데이터 전처리

        Args:
            obs: 원본 관찰

        Returns:
            전처리된 관찰 (항상 Dict 형태)
        """
        if isinstance(obs, dict):
            # 딕셔너리 형태 유지하면서 전처리
            processed = {}

            # 이미지 정규화
            if 'image' in obs:
                img = obs['image']
                if self.normalize_obs and img.dtype == np.uint8:
                    processed['image'] = img.astype(np.float32) / 255.0
                else:
                    processed['image'] = img

            # 점수 복사
            if 'score' in obs:
                processed['score'] = obs['score']

            # 다른 필드도 모두 복사
            for key, value in obs.items():
                if key not in processed:
                    processed[key] = value

        else:
            # dict가 아닌 경우, dict로 감싸기
            if self.normalize_obs and isinstance(obs, np.ndarray) and obs.dtype == np.uint8:
                processed_array = obs.astype(np.float32) / 255.0
            else:
                processed_array = obs

            processed = {'image': processed_array, 'score': np.array([0.0], dtype=np.float32)}

        return processed

    def _process_reward(self, reward: float, info: Dict[str, Any]) -> float:
        """
        보상 함수 처리 및 커스터마이징

        Args:
            reward: 원본 보상
            info: 환경에서 제공하는 추가 정보

        Returns:
            처리된 보상
        """
        # 기본: 스케일링
        processed_reward = reward * self.reward_scale

        # 여기에 커스텀 보상 로직 추가 가능
        # 예: 콤보 보너스, 생존 보너스 등

        return processed_reward

    def _extract_features(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        관찰에서 특징 벡터 추출

        Args:
            obs: 관찰 딕셔너리

        Returns:
            특징 벡터
        """
        # 기본 특징: 점수
        features = [obs.get('score', 0)]

        # 추가 특징 추출 가능
        # 예: 보드 상태, 다음 과일 정보 등

        return np.array(features, dtype=np.float32)

    def _create_mock_env(self) -> gym.Env:
        """
        개발/테스트용 모의 환경 생성

        Returns:
            모의 Gymnasium 환경
        """
        class MockSuikaEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict({
                    'image': gym.spaces.Box(
                        low=0, high=255,
                        shape=(400, 300, 3),
                        dtype=np.uint8
                    ),
                    'score': gym.spaces.Box(
                        low=0, high=np.inf,
                        shape=(1,),
                        dtype=np.float32
                    )
                })
                self.action_space = gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(1,),
                    dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                obs = {
                    'image': np.zeros((400, 300, 3), dtype=np.uint8),
                    'score': np.array([0.0], dtype=np.float32)
                }
                return obs, {}

            def step(self, action):
                obs = {
                    'image': np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8),
                    'score': np.array([np.random.randint(0, 100)], dtype=np.float32)
                }
                reward = float(np.random.randint(0, 10))
                terminated = np.random.random() < 0.01
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info

        return MockSuikaEnv()

    def get_episode_statistics(self) -> Dict[str, float]:
        """
        현재 에피소드 통계 반환

        Returns:
            통계 딕셔너리
        """
        return {
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'best_score': self.best_score,
            'average_reward': self.episode_score / max(1, self.episode_steps)
        }


def make_suika_env(
    observation_type: str = "image",
    reward_scale: float = 1.0,
    **kwargs
) -> SuikaEnvWrapper:
    """
    Suika 환경 생성 헬퍼 함수

    Args:
        observation_type: 관찰 타입
        reward_scale: 보상 스케일
        **kwargs: 추가 환경 설정

    Returns:
        설정된 Suika 환경
    """
    return SuikaEnvWrapper(
        observation_type=observation_type,
        reward_scale=reward_scale,
        **kwargs
    )
