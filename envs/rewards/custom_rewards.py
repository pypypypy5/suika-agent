"""
커스텀 보상 함수 예제

다양한 보상 전략을 구현한 예제 함수들입니다.
개발자는 이를 참고하여 자신만의 보상 함수를 만들 수 있습니다.
"""

from typing import Dict, Any
from .base import BaseRewardFunction


class SurvivalBonusReward(BaseRewardFunction):
    """
    생존 보너스를 추가하는 보상 함수

    점수 증가분에 더해 매 스텝마다 생존 보너스를 제공합니다.
    이는 에이전트가 가능한 오래 생존하도록 유도합니다.

    Attributes:
        score_weight: 점수 증가분에 대한 가중치
        survival_bonus: 매 스텝마다 제공되는 생존 보너스
    """

    def __init__(self, score_weight: float = 1.0, survival_bonus: float = 0.01):
        """
        Args:
            score_weight: 점수 증가분에 곱할 가중치
            survival_bonus: 매 스텝마다 추가할 생존 보너스
        """
        self.score_weight = score_weight
        self.survival_bonus = survival_bonus

    def calculate(
        self,
        observation: Dict[str, Any],
        raw_reward: float,
        info: Dict[str, Any]
    ) -> float:
        """
        점수 증가분 * 가중치 + 생존 보너스

        Args:
            observation: 현재 관찰
            raw_reward: 원본 보상 (점수 증가분)
            info: 추가 정보

        Returns:
            점수 보상 + 생존 보너스
        """
        return raw_reward * self.score_weight + self.survival_bonus


class ComboReward(BaseRewardFunction):
    """
    연속 점수 획득 시 보너스를 주는 보상 함수

    연속으로 점수를 획득하면 콤보 카운터가 증가하고,
    콤보에 비례한 보너스 배율이 적용됩니다.
    점수를 획득하지 못하면 콤보가 리셋됩니다.

    Attributes:
        base_scale: 기본 보상 스케일
        combo_bonus: 콤보당 추가되는 보너스 비율
        max_multiplier: 최대 보너스 배율
        combo_count: 현재 콤보 카운터
    """

    def __init__(
        self,
        base_scale: float = 1.0,
        combo_bonus: float = 0.1,
        max_multiplier: float = 2.0
    ):
        """
        Args:
            base_scale: 기본 보상 스케일링 팩터
            combo_bonus: 콤보당 추가되는 보너스 비율 (예: 0.1 = 10%)
            max_multiplier: 최대 콤보 배율 제한
        """
        self.base_scale = base_scale
        self.combo_bonus = combo_bonus
        self.max_multiplier = max_multiplier
        self.combo_count = 0

    def calculate(
        self,
        observation: Dict[str, Any],
        raw_reward: float,
        info: Dict[str, Any]
    ) -> float:
        """
        콤보를 고려한 보상 계산

        점수가 증가했으면 콤보 증가, 아니면 콤보 리셋.
        콤보에 따른 배율을 적용하여 보상 계산.

        Args:
            observation: 현재 관찰
            raw_reward: 원본 보상 (점수 증가분)
            info: 추가 정보

        Returns:
            콤보 배율이 적용된 보상
        """
        # 점수가 증가했으면 콤보 증가, 아니면 리셋
        if raw_reward > 0:
            self.combo_count += 1
        else:
            self.combo_count = 0

        # 콤보에 따른 보너스 배율 계산
        multiplier = 1.0 + (self.combo_count * self.combo_bonus)
        multiplier = min(multiplier, self.max_multiplier)

        return raw_reward * self.base_scale * multiplier

    def reset(self) -> None:
        """에피소드 시작 시 콤보 카운터 초기화"""
        self.combo_count = 0


class ShapedReward(BaseRewardFunction):
    """
    복합적인 보상 shaping을 적용하는 함수

    점수, 생존, 그리고 게임 상태에 따른 다양한 보너스/페널티를 결합합니다.
    이는 더 세밀한 학습 신호를 제공하여 학습 효율을 높일 수 있습니다.

    Attributes:
        score_weight: 점수 증가분 가중치
        survival_bonus: 생존 보너스
        step_penalty: 스텝당 페널티 (음수값)
    """

    def __init__(
        self,
        score_weight: float = 1.0,
        survival_bonus: float = 0.01,
        step_penalty: float = -0.001
    ):
        """
        Args:
            score_weight: 점수 증가분에 대한 가중치
            survival_bonus: 매 스텝 생존 보너스
            step_penalty: 매 스텝 페널티 (빠른 게임 종료 유도)
        """
        self.score_weight = score_weight
        self.survival_bonus = survival_bonus
        self.step_penalty = step_penalty

    def calculate(
        self,
        observation: Dict[str, Any],
        raw_reward: float,
        info: Dict[str, Any]
    ) -> float:
        """
        복합 보상 계산

        Args:
            observation: 현재 관찰
            raw_reward: 원본 보상 (점수 증가분)
            info: 추가 정보

        Returns:
            복합 보상 (점수 + 보너스 + 페널티)
        """
        total_reward = raw_reward * self.score_weight

        # 생존 보너스
        total_reward += self.survival_bonus

        # 스텝 페널티 (효율적인 플레이 유도)
        total_reward += self.step_penalty

        return total_reward
