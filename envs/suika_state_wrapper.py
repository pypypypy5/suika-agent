"""
Suika Game 상태 기반 환경 래퍼

이미지 대신 게임 상태(과일 위치, 크기 등)를 직접 관찰로 제공합니다.
CNN 없이도 학습 가능하며, 훨씬 효율적입니다.
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


class SuikaStateWrapper(gym.Wrapper):
    """
    Suika Game 상태 기반 래퍼

    이미지 대신 구조화된 게임 상태를 제공:
    - 각 과일의 위치 (x, y)
    - 각 과일의 크기/타입
    - 다음 과일 타입
    - 현재 점수

    이를 통해 CNN 없이도 MLP로 학습 가능하며, 훨씬 빠르고 효율적입니다.
    """

    def __init__(
        self,
        headless: bool = True,
        port: int = 8923,
        delay_before_img_capture: float = 0.5,
        max_fruits: int = 50,  # 관찰 벡터 크기 고정을 위한 최대 과일 수
        reward_scale: float = 1.0,
        use_mock: bool = False,
        **kwargs
    ):
        """
        Args:
            headless: Selenium 브라우저 headless 모드
            port: 로컬 HTTP 서버 포트
            delay_before_img_capture: 이미지 캡처 전 대기 시간
            max_fruits: 관찰 벡터에 포함할 최대 과일 수
            reward_scale: 보상 스케일링 팩터
            use_mock: Mock 환경 사용 여부
        """
        if use_mock:
            print("Using mock environment for development/testing.")
            base_env = self._create_mock_env()
        else:
            try:
                from suika_env.suika_browser_env import SuikaBrowserEnv
                base_env = SuikaBrowserEnv(
                    headless=headless,
                    port=port,
                    delay_before_img_capture=delay_before_img_capture
                )
                print(f"Using real Suika environment with state extraction (headless={headless})")
            except Exception as e:
                print(f"Warning: Could not create Suika environment: {e}")
                print("Falling back to mock environment.")
                base_env = self._create_mock_env()

        super().__init__(base_env)

        self.max_fruits = max_fruits
        self.reward_scale = reward_scale

        # 관찰 공간 재정의: 구조화된 상태
        # [next_fruit_type(1), score(1), fruit_states(max_fruits * 3)]
        # fruit_state: [x, y, type] for each fruit
        obs_size = 2 + (max_fruits * 3)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )

        # 에피소드 통계
        self.episode_score = 0
        self.episode_steps = 0
        self.best_score = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """환경 리셋"""
        obs, info = self.env.reset(seed=seed, options=options)

        # 에피소드 통계 초기화
        self.episode_score = 0
        self.episode_steps = 0

        # 상태 기반 관찰로 변환
        state_obs = self._extract_state(obs)

        info.update({
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'best_score': self.best_score
        })

        return state_obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """환경 스텝"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 에피소드 통계 업데이트
        self.episode_steps += 1
        self.episode_score += reward

        # 보상 처리
        processed_reward = reward * self.reward_scale

        # 상태 기반 관찰로 변환
        state_obs = self._extract_state(obs)

        info.update({
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'original_reward': reward,
            'processed_reward': processed_reward
        })

        if terminated or truncated:
            if self.episode_score > self.best_score:
                self.best_score = self.episode_score
            info['best_score'] = self.best_score

        return state_obs, processed_reward, terminated, truncated, info

    def _extract_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        이미지 관찰을 구조화된 상태로 변환

        Returns:
            np.ndarray: [next_fruit, score, fruit1_x, fruit1_y, fruit1_type, ...]
        """
        # Mock 환경의 game_state 확인
        if isinstance(obs, dict) and '_game_state' in obs:
            game_state = obs['_game_state']
            return self._encode_state(game_state)

        # 실제 환경인 경우 JavaScript로부터 상태 추출
        if hasattr(self.env, 'driver'):
            try:
                game_state = self._get_game_state_from_js()
                return self._encode_state(game_state)
            except Exception as e:
                print(f"Warning: Failed to extract game state: {e}")
                # Fallback: 점수만 사용
                return self._encode_fallback_state(obs)
        else:
            # Fallback
            return self._encode_fallback_state(obs)

    def _get_game_state_from_js(self) -> Dict[str, Any]:
        """
        JavaScript로부터 게임 상태를 직접 추출

        Returns:
            {
                'next_fruit': int (0-10),
                'score': float,
                'fruits': [{'x': float, 'y': float, 'type': int}, ...]
            }
        """
        js_code = """
        // Matter.js world에서 모든 과일 정보 추출
        const fruits = [];
        const bodies = Composite.allBodies(engine.world);

        for (const body of bodies) {
            // 과일인지 확인 (label이 'fruit'이거나 sizeIndex가 있는 경우)
            if (body.label && body.label.startsWith('fruit')) {
                const sizeIndex = parseInt(body.label.split('-')[1]) || 0;
                fruits.push({
                    x: body.position.x,
                    y: body.position.y,
                    type: sizeIndex,
                    radius: body.circleRadius || 0
                });
            }
        }

        return {
            next_fruit: window.Game.nextFruitSize,
            score: window.Game.score,
            fruits: fruits,
            game_width: window.Game.width,
            game_height: window.Game.height
        };
        """

        try:
            result = self.env.driver.execute_script(js_code)
            return result
        except Exception as e:
            print(f"JS execution failed: {e}")
            # Fallback
            return {
                'next_fruit': 0,
                'score': obs.get('score', [0])[0] if isinstance(obs, dict) else 0,
                'fruits': [],
                'game_width': 640,
                'game_height': 960
            }

    def _encode_state(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        게임 상태를 고정 크기 벡터로 인코딩

        Args:
            game_state: JavaScript에서 추출한 게임 상태

        Returns:
            np.ndarray: 고정 크기 상태 벡터
        """
        state_vector = []

        # 1. 다음 과일 타입 (정규화: 0-10 → -1 to 1)
        next_fruit = game_state.get('next_fruit', 0)
        state_vector.append((next_fruit / 5.0) - 1.0)

        # 2. 점수 (정규화: 0-10000 → -1 to 1)
        score = game_state.get('score', 0)
        state_vector.append(min(score / 5000.0, 1.0) * 2 - 1.0)

        # 3. 과일 정보
        fruits = game_state.get('fruits', [])
        game_width = game_state.get('game_width', 640)
        game_height = game_state.get('game_height', 960)

        # 최대 개수까지만 사용 (오래된 과일부터, 또는 y 좌표 기준 정렬)
        fruits_sorted = sorted(fruits, key=lambda f: f.get('y', 0))[:self.max_fruits]

        for i in range(self.max_fruits):
            if i < len(fruits_sorted):
                fruit = fruits_sorted[i]
                # x 좌표 정규화 (0-640 → -1 to 1)
                x_norm = (fruit.get('x', 0) / game_width) * 2 - 1.0
                # y 좌표 정규화 (0-960 → -1 to 1)
                y_norm = (fruit.get('y', 0) / game_height) * 2 - 1.0
                # 타입 정규화 (0-10 → -1 to 1)
                type_norm = (fruit.get('type', 0) / 5.0) - 1.0

                state_vector.extend([x_norm, y_norm, type_norm])
            else:
                # 패딩: 과일이 없으면 0으로 채움
                state_vector.extend([0.0, 0.0, 0.0])

        return np.array(state_vector, dtype=np.float32)

    def _encode_fallback_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Fallback: 점수만 사용하는 간단한 상태 인코딩
        """
        state_vector = []

        # 다음 과일: 랜덤
        state_vector.append(0.0)

        # 점수
        if isinstance(obs, dict) and 'score' in obs:
            score = obs['score'][0] if hasattr(obs['score'], '__getitem__') else obs['score']
            state_vector.append(min(score / 5000.0, 1.0) * 2 - 1.0)
        else:
            state_vector.append(0.0)

        # 과일 정보: 모두 0
        state_vector.extend([0.0] * (self.max_fruits * 3))

        return np.array(state_vector, dtype=np.float32)

    def _create_mock_env(self) -> gym.Env:
        """Mock 환경 생성 (상태 추출 테스트용)"""
        class MockSuikaEnvWithState(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = gym.spaces.Dict({
                    'image': gym.spaces.Box(low=0, high=255, shape=(400, 300, 3), dtype=np.uint8),
                    'score': gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
                })
                self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                self.fruits = []
                self.score = 0
                self.next_fruit = 0

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                self.fruits = []
                self.score = 0
                self.next_fruit = np.random.randint(0, 5)

                obs = {
                    'image': np.zeros((400, 300, 3), dtype=np.uint8),
                    'score': np.array([self.score], dtype=np.float32),
                    # Mock 상태 추가
                    '_game_state': {
                        'next_fruit': self.next_fruit,
                        'score': self.score,
                        'fruits': self.fruits,
                        'game_width': 640,
                        'game_height': 960
                    }
                }
                return obs, {}

            def step(self, action):
                # 과일 추가 시뮬레이션
                x = action[0] * 640
                y = len(self.fruits) * 50  # 쌓이는 효과
                fruit_type = np.random.randint(0, 5)

                self.fruits.append({
                    'x': x,
                    'y': y,
                    'type': fruit_type
                })

                # 최대 10개까지만 유지
                if len(self.fruits) > 10:
                    self.fruits.pop(0)

                # 점수 증가
                reward = float(np.random.randint(1, 10))
                self.score += reward
                self.next_fruit = np.random.randint(0, 5)

                obs = {
                    'image': np.random.randint(0, 256, (400, 300, 3), dtype=np.uint8),
                    'score': np.array([self.score], dtype=np.float32),
                    '_game_state': {
                        'next_fruit': self.next_fruit,
                        'score': self.score,
                        'fruits': self.fruits,
                        'game_width': 640,
                        'game_height': 960
                    }
                }

                terminated = np.random.random() < 0.01
                truncated = False
                info = {}
                return obs, reward, terminated, truncated, info

        return MockSuikaEnvWithState()

    def get_episode_statistics(self) -> Dict[str, float]:
        """에피소드 통계 반환"""
        return {
            'episode_score': self.episode_score,
            'episode_steps': self.episode_steps,
            'best_score': self.best_score,
            'average_reward': self.episode_score / max(1, self.episode_steps)
        }
