"""
보상 함수 기본 클래스

모든 커스텀 보상 함수는 이 클래스를 상속받아 구현해야 합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRewardFunction(ABC):
    """
    보상 함수의 추상 기본 클래스

    모든 보상 함수는 이 클래스를 상속받아 calculate() 메서드를 구현해야 합니다.
    """

    @abstractmethod
    def calculate(
        self,
        observation: Dict[str, Any],
        raw_reward: float,
        info: Dict[str, Any]
    ) -> float:
        """
        관찰, 원본 보상, 추가 정보를 바탕으로 최종 보상 계산

        Args:
            observation: 현재 관찰 (image, score 등)
            raw_reward: 환경에서 제공한 원본 보상 (점수 증가분)
            info: 추가 메타데이터

        Returns:
            처리된 최종 보상값
        """
        pass

    def reset(self) -> None:
        """
        에피소드 시작 시 내부 상태 초기화

        상태를 유지하는 보상 함수(예: 콤보 카운터)는 이 메서드를 오버라이드하여
        에피소드마다 상태를 초기화해야 합니다.
        """
        pass
