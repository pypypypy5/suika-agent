"""
학습 트레이너

에이전트 학습 루프를 관리하는 트레이너 클래스
"""

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm

from agents.base_agent import BaseAgent
from envs.suika_wrapper import SuikaEnvWrapper
from utils.logger import Logger


class Trainer:
    """
    강화학습 에이전트 학습 트레이너

    환경과 에이전트 간의 상호작용을 관리하고
    학습 프로세스를 조율합니다.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: SuikaEnvWrapper,
        logger: Logger,
        config: Dict[str, Any]
    ):
        """
        Args:
            agent: 학습할 에이전트
            env: 학습 환경
            logger: 로거
            config: 학습 설정
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.config = config

        # 학습 설정
        training_config = config.get('training', {})
        self.total_timesteps = training_config.get('total_timesteps', 1000000)
        self.eval_freq = training_config.get('eval_freq', 10000)
        self.eval_episodes = training_config.get('eval_episodes', 10)
        self.save_freq = training_config.get('save_freq', 50000)
        self.log_interval = config.get('logging', {}).get('log_interval', 100)

        # 체크포인트 경로
        self.save_path = Path(training_config.get('save_path', 'experiments/checkpoints'))
        self.save_path.mkdir(parents=True, exist_ok=True)

        # 통계
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf

    def train(self) -> None:
        """
        에이전트 학습 실행
        """
        print("=" * 50)
        print("Starting Training")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print("=" * 50)

        # 에이전트를 학습 모드로
        self.agent.train()

        # 환경 초기화
        obs, info = self.env.reset()

        episode_reward = 0
        episode_length = 0
        episode_num = 0

        # 학습 루프
        with tqdm(total=self.total_timesteps, desc="Training") as pbar:
            for step in range(self.total_timesteps):
                # 행동 선택
                action = self.agent.select_action(obs, deterministic=False)

                # 환경 스텝
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 에이전트 업데이트 (구체적인 에이전트 구현에 따라 다름)
                # 예: 리플레이 버퍼에 저장하고 주기적으로 학습
                update_info = self.agent.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done
                )

                # 통계 업데이트
                episode_reward += reward
                episode_length += 1
                self.agent.total_steps += 1

                # 에피소드 종료
                if done:
                    episode_num += 1
                    self.agent.episodes += 1

                    # 에피소드 통계 기록
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    # 로깅
                    self.logger.log_episode(
                        episode=episode_num,
                        episode_reward=episode_reward,
                        episode_length=episode_length,
                        additional_metrics={
                            'best_score': info.get('best_score', 0)
                        }
                    )

                    # 콘솔 출력
                    if episode_num % self.log_interval == 0:
                        mean_reward = np.mean(self.episode_rewards[-100:])
                        mean_length = np.mean(self.episode_lengths[-100:])
                        self.logger.print_progress(
                            step=step,
                            total_steps=self.total_timesteps,
                            metrics={
                                'episode': episode_num,
                                'mean_reward': mean_reward,
                                'mean_length': mean_length
                            }
                        )

                    # 환경 리셋
                    obs, info = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs

                # 학습 메트릭 로깅
                if update_info and step % self.log_interval == 0:
                    self.logger.log_training(step, update_info.get('loss', 0), update_info)

                # 평가
                if step > 0 and step % self.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    mean_eval_reward = eval_metrics['mean_reward']

                    print(f"\nEvaluation at step {step}:")
                    print(f"  Mean reward: {mean_eval_reward:.2f}")
                    print(f"  Mean length: {eval_metrics['mean_length']:.2f}")

                    # 최고 성능 모델 저장
                    if mean_eval_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_eval_reward
                        self.save_checkpoint('best_model.pth')
                        print(f"  New best model saved! (reward: {mean_eval_reward:.2f})")

                # 정기 체크포인트 저장
                if step > 0 and step % self.save_freq == 0:
                    self.save_checkpoint(f'checkpoint_{step}.pth')

                pbar.update(1)

        # 학습 종료
        print("\nTraining completed!")
        self.save_checkpoint('final_model.pth')
        self.logger.close()

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        에이전트 평가

        Args:
            num_episodes: 평가 에피소드 수 (None이면 설정값 사용)

        Returns:
            평가 메트릭
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        # 평가 모드로 전환
        self.agent.eval()

        eval_rewards = []
        eval_lengths = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # 결정적 행동 선택
                action = self.agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)

        # 학습 모드로 복귀
        self.agent.train()

        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }

        return metrics

    def save_checkpoint(self, filename: str) -> None:
        """
        체크포인트 저장

        Args:
            filename: 저장 파일명
        """
        path = self.save_path / filename
        self.agent.save(str(path))

    def load_checkpoint(self, filename: str) -> None:
        """
        체크포인트 로드

        Args:
            filename: 로드 파일명
        """
        path = self.save_path / filename
        self.agent.load(str(path))
