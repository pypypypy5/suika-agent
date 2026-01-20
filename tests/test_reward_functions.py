"""
Reward 함수 유닛 테스트

각 reward 함수가 올바르게 동작하는지 독립적으로 테스트합니다.
"""

import pytest
import numpy as np
from envs.rewards import (
    BaseRewardFunction,
    ScoreBasedReward,
    SurvivalBonusReward,
    ComboReward,
    ShapedReward
)


class TestScoreBasedReward:
    """기본 점수 기반 보상 함수 테스트"""

    def test_default_scale(self):
        """기본 스케일(1.0)로 보상 계산"""
        reward_fn = ScoreBasedReward(scale=1.0)
        result = reward_fn.calculate({}, 10.0, {})
        assert result == 10.0

    def test_custom_scale(self):
        """커스텀 스케일로 보상 계산"""
        reward_fn = ScoreBasedReward(scale=2.0)
        result = reward_fn.calculate({}, 10.0, {})
        assert result == 20.0

    def test_zero_reward(self):
        """0 보상 처리"""
        reward_fn = ScoreBasedReward(scale=1.5)
        result = reward_fn.calculate({}, 0.0, {})
        assert result == 0.0

    def test_negative_reward(self):
        """음수 보상 처리 (있을 수 있음)"""
        reward_fn = ScoreBasedReward(scale=1.0)
        result = reward_fn.calculate({}, -5.0, {})
        assert result == -5.0


class TestSurvivalBonusReward:
    """생존 보너스 보상 함수 테스트"""

    def test_with_score(self):
        """점수와 생존 보너스 합산"""
        reward_fn = SurvivalBonusReward(score_weight=1.0, survival_bonus=0.1)
        result = reward_fn.calculate({}, 5.0, {})
        assert result == 5.1

    def test_without_score(self):
        """점수 없이 생존 보너스만"""
        reward_fn = SurvivalBonusReward(score_weight=1.0, survival_bonus=0.1)
        result = reward_fn.calculate({}, 0.0, {})
        assert result == 0.1

    def test_custom_weights(self):
        """커스텀 가중치 테스트"""
        reward_fn = SurvivalBonusReward(score_weight=2.0, survival_bonus=0.5)
        result = reward_fn.calculate({}, 10.0, {})
        assert result == 20.5  # 10 * 2.0 + 0.5


class TestComboReward:
    """콤보 보상 함수 테스트"""

    def test_first_score(self):
        """첫 번째 점수 (콤보 1)"""
        reward_fn = ComboReward(base_scale=1.0, combo_bonus=0.1)
        result = reward_fn.calculate({}, 10.0, {})
        # 콤보 1 -> multiplier = 1.0 + (1 * 0.1) = 1.1
        assert result == 11.0

    def test_combo_accumulation(self):
        """연속 점수로 콤보 누적"""
        reward_fn = ComboReward(base_scale=1.0, combo_bonus=0.1)

        # 첫 번째 점수: 콤보 1
        r1 = reward_fn.calculate({}, 10.0, {})
        assert r1 == 11.0  # 10 * 1.1

        # 두 번째 점수: 콤보 2
        r2 = reward_fn.calculate({}, 10.0, {})
        assert r2 == 12.0  # 10 * 1.2

        # 세 번째 점수: 콤보 3
        r3 = reward_fn.calculate({}, 10.0, {})
        assert r3 == 13.0  # 10 * 1.3

    def test_combo_reset(self):
        """점수 없으면 콤보 리셋"""
        reward_fn = ComboReward(base_scale=1.0, combo_bonus=0.1)

        # 콤보 쌓기
        reward_fn.calculate({}, 10.0, {})
        reward_fn.calculate({}, 10.0, {})
        assert reward_fn.combo_count == 2

        # 점수 없음 -> 콤보 리셋
        result = reward_fn.calculate({}, 0.0, {})
        assert result == 0.0
        assert reward_fn.combo_count == 0

    def test_max_multiplier(self):
        """최대 배율 제한"""
        reward_fn = ComboReward(base_scale=1.0, combo_bonus=0.1, max_multiplier=1.5)

        # 콤보를 10번 쌓음 (multiplier가 2.0까지 갈 수 있음)
        for _ in range(10):
            result = reward_fn.calculate({}, 10.0, {})

        # 최대 배율은 1.5로 제한됨
        assert result <= 15.0  # 10 * 1.5

    def test_reset_method(self):
        """reset() 메서드로 상태 초기화"""
        reward_fn = ComboReward(base_scale=1.0)

        # 콤보 쌓기
        reward_fn.calculate({}, 10.0, {})
        reward_fn.calculate({}, 10.0, {})
        assert reward_fn.combo_count > 0

        # 리셋
        reward_fn.reset()
        assert reward_fn.combo_count == 0


class TestShapedReward:
    """복합 보상 함수 테스트"""

    def test_all_components(self):
        """점수, 생존 보너스, 스텝 페널티 모두 적용"""
        reward_fn = ShapedReward(
            score_weight=1.0,
            survival_bonus=0.01,
            step_penalty=-0.001
        )

        result = reward_fn.calculate({}, 10.0, {})
        expected = 10.0 * 1.0 + 0.01 - 0.001
        assert result == expected

    def test_no_score_step(self):
        """점수 없는 스텝"""
        reward_fn = ShapedReward(
            score_weight=1.0,
            survival_bonus=0.01,
            step_penalty=-0.001
        )

        result = reward_fn.calculate({}, 0.0, {})
        expected = 0.0 + 0.01 - 0.001
        assert result == expected


class TestRewardFunctionInterface:
    """Reward 함수 인터페이스 테스트"""

    def test_base_class_is_abstract(self):
        """BaseRewardFunction은 추상 클래스"""
        with pytest.raises(TypeError):
            BaseRewardFunction()

    def test_custom_reward_implementation(self):
        """커스텀 reward 함수 구현 가능"""

        class CustomReward(BaseRewardFunction):
            def calculate(self, observation, raw_reward, info):
                return raw_reward * 3.0

        reward_fn = CustomReward()
        result = reward_fn.calculate({}, 10.0, {})
        assert result == 30.0

    def test_reset_default_behavior(self):
        """reset()의 기본 동작 (아무것도 안 함)"""
        reward_fn = ScoreBasedReward()
        # reset()이 에러 없이 실행되어야 함
        reward_fn.reset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
