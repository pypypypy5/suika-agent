"""
Wrapper와 Reward 함수 통합 테스트

SuikaEnvWrapper가 reward 함수와 올바르게 통합되는지 테스트합니다.
"""

import pytest
import numpy as np
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import SuikaEnvWrapper
from envs.rewards import (
    ScoreBasedReward,
    ComboReward,
    SurvivalBonusReward,
    BaseRewardFunction
)


class TestWrapperWithDefaultReward:
    """기본 reward 함수를 사용한 wrapper 테스트"""

    def test_backward_compatibility_with_reward_scale(self):
        """하위 호환성: reward_scale만 사용"""
        env = SuikaEnvWrapper(use_mock=True, reward_scale=2.0)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # reward_scale이 적용되어야 함
        original_reward = info.get('original_reward', 0)
        if original_reward != 0:  # Mock 환경에서 0이 아닌 보상이 나온 경우
            assert reward == pytest.approx(original_reward * 2.0)

        env.close()

    def test_default_reward_fn_creation(self):
        """reward_fn이 None이면 기본 ScoreBasedReward 생성"""
        env = SuikaEnvWrapper(use_mock=True, reward_scale=1.5)

        # reward_fn이 ScoreBasedReward 인스턴스여야 함
        assert isinstance(env.reward_fn, ScoreBasedReward)
        assert env.reward_fn.scale == 1.5

        env.close()

    def test_basic_step_execution(self):
        """기본 step 실행 테스트"""
        env = SuikaEnvWrapper(use_mock=True)

        obs, info = env.reset()
        assert 'episode_score' in info
        assert 'episode_steps' in info

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert 'original_reward' in info
        assert 'processed_reward' in info
        assert isinstance(reward, (int, float))

        env.close()


class TestWrapperWithCustomReward:
    """커스텀 reward 함수를 사용한 wrapper 테스트"""

    def test_combo_reward_integration(self):
        """ComboReward 함수 통합 테스트"""
        reward_fn = ComboReward(base_scale=1.0, combo_bonus=0.1)
        env = SuikaEnvWrapper(use_mock=True, reward_fn=reward_fn)

        obs, info = env.reset()
        assert reward_fn.combo_count == 0  # reset 후 콤보 초기화

        # 여러 스텝 실행
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            assert 'processed_reward' in info
            if terminated:
                break

        env.close()

    def test_survival_bonus_reward_integration(self):
        """SurvivalBonusReward 함수 통합 테스트"""
        reward_fn = SurvivalBonusReward(score_weight=1.0, survival_bonus=0.05)
        env = SuikaEnvWrapper(use_mock=True, reward_fn=reward_fn)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 생존 보너스가 포함되어야 함
        original_reward = info.get('original_reward', 0)
        expected_reward = original_reward * 1.0 + 0.05
        assert reward == pytest.approx(expected_reward)

        env.close()

    def test_custom_reward_with_observation(self):
        """관찰 정보를 사용하는 커스텀 reward 함수"""

        class ObservationBasedReward(BaseRewardFunction):
            def calculate(self, observation, raw_reward, info):
                # 관찰에서 점수 정보를 사용
                if 'score' in observation:
                    score_value = observation['score'][0] if hasattr(observation['score'], '__getitem__') else observation['score']
                    # 점수가 높을수록 보너스
                    bonus = score_value * 0.001
                    return raw_reward + bonus
                return raw_reward

        reward_fn = ObservationBasedReward()
        env = SuikaEnvWrapper(use_mock=True, reward_fn=reward_fn)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # reward 함수가 실행되었는지 확인
        assert 'processed_reward' in info

        env.close()


class TestRewardFunctionReset:
    """Reward 함수의 reset 동작 테스트"""

    def test_combo_reset_on_episode_reset(self):
        """에피소드 reset 시 콤보 초기화"""
        reward_fn = ComboReward(base_scale=1.0)
        env = SuikaEnvWrapper(use_mock=True, reward_fn=reward_fn)

        # 첫 번째 에피소드
        obs, info = env.reset()
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # 콤보가 쌓였을 수 있음
        combo_before_reset = reward_fn.combo_count

        # 에피소드 리셋
        obs, info = env.reset()

        # 콤보가 초기화되어야 함
        assert reward_fn.combo_count == 0

        env.close()


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_zero_reward_handling(self):
        """0 보상 처리"""
        env = SuikaEnvWrapper(use_mock=True, reward_scale=2.0)

        obs, info = env.reset()
        # Mock 환경은 랜덤 보상을 주지만, 0이 나올 수도 있음
        # 이 테스트는 0 보상이 와도 크래시하지 않는지 확인

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert isinstance(reward, (int, float))
            if terminated:
                break

        env.close()

    def test_multiple_episodes(self):
        """여러 에피소드 실행"""
        reward_fn = ComboReward(base_scale=1.0)
        env = SuikaEnvWrapper(use_mock=True, reward_fn=reward_fn)

        for episode in range(3):
            obs, info = env.reset()
            assert reward_fn.combo_count == 0  # 매 에피소드 초기화

            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    break

        env.close()

    def test_info_dict_structure(self):
        """info 딕셔너리 구조 검증"""
        env = SuikaEnvWrapper(use_mock=True)

        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # 필수 키 확인
        required_keys = ['episode_score', 'episode_steps', 'original_reward', 'processed_reward']
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

        env.close()


class TestRewardScaleDeprecation:
    """reward_scale과 reward_fn의 상호작용 테스트"""

    def test_reward_fn_overrides_reward_scale(self):
        """reward_fn이 주어지면 reward_scale은 무시됨"""
        custom_reward = ScoreBasedReward(scale=3.0)
        env = SuikaEnvWrapper(
            use_mock=True,
            reward_scale=2.0,  # 이 값은 무시되어야 함
            reward_fn=custom_reward
        )

        # reward_fn이 custom_reward여야 함
        assert env.reward_fn is custom_reward
        assert env.reward_fn.scale == 3.0  # reward_scale=2.0이 아닌 3.0

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
