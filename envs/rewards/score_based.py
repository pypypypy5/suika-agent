"""
점수 기반 보상 함수

가장 기본적인 보상 함수로, 점수 증가분을 그대로 보상으로 사용합니다.
이는 기존 환경의 기본 동작을 유지합니다.
"""

from typing import Dict, Any
from .base import BaseRewardFunction


class ScoreBasedReward(BaseRewardFunction):
    """
    기본 점수 기반 보상 함수

    환경에서 제공하는 점수 증가분을 스케일링하여 보상으로 사용합니다.
    이는 기존 환경의 기본 동작과 동일합니다.

    Attributes:
        scale: 보상 스케일링 팩터 (기본값: 1.0)
    """

    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: 보상 값에 곱할 스케일링 팩터
        """
        self.scale = scale

    def calculate(
        self,
        observation: Dict[str, Any],
        raw_reward: float,
        info: Dict[str, Any]
    ) -> float:
        """
        점수 증가분 * 스케일로 보상 계산

        Args:
            observation: 현재 관찰 (사용 안 함)
            raw_reward: 환경에서 제공한 원본 보상 (점수 증가분)
            info: 추가 정보 (사용 안 함)

        Returns:
            스케일링된 보상
        """
        return raw_reward * self.scale
