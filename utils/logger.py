"""
로깅 유틸리티

학습 과정의 메트릭과 진행상황을 로깅하는 유틸리티
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime


class Logger:
    """
    학습 로거 클래스

    TensorBoard, WandB, 콘솔 로깅을 통합 관리
    """

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Args:
            log_dir: 로그 저장 디렉토리
            use_tensorboard: TensorBoard 사용 여부
            use_wandb: Weights & Biases 사용 여부
            wandb_project: WandB 프로젝트 이름
            experiment_name: 실험 이름
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 실험 이름 설정
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # TensorBoard 설정
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / experiment_name
                self.tb_writer = SummaryWriter(str(tb_dir))
                print(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                print("Warning: tensorboard not installed. Skipping TensorBoard logging.")
                self.use_tensorboard = False

        # WandB 설정
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project or "suika-rl",
                    name=experiment_name,
                    dir=str(self.log_dir)
                )
                print(f"WandB logging initialized for project {wandb_project}")
            except ImportError:
                print("Warning: wandb not installed. Skipping WandB logging.")
                self.use_wandb = False

        # 메트릭 저장
        self.metrics_history: Dict[str, list] = {}

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        스칼라 값 로깅

        Args:
            tag: 메트릭 이름
            value: 값
            step: 스텝 번호
        """
        # 히스토리 저장
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append((step, value))

        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

        # WandB
        if self.use_wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """
        여러 스칼라 값 로깅

        Args:
            metrics: 메트릭 딕셔너리
            step: 스텝 번호
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """
        히스토그램 로깅

        Args:
            tag: 메트릭 이름
            values: 값 배열
            step: 스텝 번호
        """
        # TensorBoard
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

        # WandB
        if self.use_wandb:
            import wandb
            wandb.log({tag: wandb.Histogram(values)}, step=step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        에피소드 결과 로깅

        Args:
            episode: 에피소드 번호
            episode_reward: 에피소드 총 보상
            episode_length: 에피소드 길이
            additional_metrics: 추가 메트릭
        """
        metrics = {
            'episode/reward': episode_reward,
            'episode/length': episode_length
        }

        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f'episode/{key}'] = value

        self.log_scalars(metrics, episode)

    def log_training(
        self,
        step: int,
        loss: float,
        additional_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        학습 메트릭 로깅

        Args:
            step: 학습 스텝
            loss: 손실 값
            additional_metrics: 추가 메트릭
        """
        metrics = {'training/loss': loss}

        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f'training/{key}'] = value

        self.log_scalars(metrics, step)

    def print_progress(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        진행상황 콘솔 출력

        Args:
            step: 현재 스텝
            total_steps: 총 스텝
            metrics: 출력할 메트릭
        """
        progress = (step / total_steps) * 100
        metrics_str = " | ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
        print(f"[{progress:.1f}%] Step {step}/{total_steps} | {metrics_str}")

    def save_metrics(self, filename: str = "metrics.npz") -> None:
        """
        메트릭 히스토리 저장

        Args:
            filename: 저장 파일명
        """
        save_path = self.log_dir / filename
        np.savez(
            save_path,
            **{key: np.array(values) for key, values in self.metrics_history.items()}
        )
        print(f"Metrics saved to {save_path}")

    def close(self) -> None:
        """로거 종료 및 리소스 정리"""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            import wandb
            wandb.finish()

        # 메트릭 저장
        self.save_metrics()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logger(config: Dict[str, Any], experiment_name: Optional[str] = None) -> Logger:
    """
    설정에서 로거 생성

    Args:
        config: 설정 딕셔너리
        experiment_name: 실험 이름

    Returns:
        설정된 Logger 인스턴스
    """
    log_config = config.get('logging', {})

    logger = Logger(
        log_dir=config.get('training', {}).get('save_path', 'experiments'),
        use_tensorboard=log_config.get('use_tensorboard', True),
        use_wandb=log_config.get('use_wandb', False),
        wandb_project=log_config.get('wandb_project', 'suika-rl'),
        experiment_name=experiment_name
    )

    return logger
