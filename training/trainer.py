"""
학습 트레이너

에이전트 학습 루프를 관리하는 트레이너 클래스
"""

from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

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
        에이전트 학습 실행 (VectorEnv 지원)
        """
        # VectorEnv 감지
        num_envs = getattr(self.env, 'num_envs', 1)

        print("=" * 50)
        print("Starting Training")
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Agent: {self.agent.__class__.__name__}")
        print(f"Num environments: {num_envs}")
        print("=" * 50)

        # 에이전트를 학습 모드로
        self.agent.train()

        # 환경 초기화
        obs, info = self.env.reset()

        # 환경별 통계 (VectorEnv 대응)
        episode_rewards = [0.0] * num_envs
        episode_lengths = [0] * num_envs
        episode_num = 0

        # 학습 루프
        # total_timesteps는 총 environment interaction 수 (총 sample 수)
        # num_envs개 환경이 병렬로 실행되므로, 실제 loop 횟수는 total_timesteps / num_envs
        total_steps = 0
        with tqdm(total=self.total_timesteps, desc="Training") as pbar:
            while total_steps < self.total_timesteps:
                # 행동 선택 (배치)
                actions = self.agent.select_action(obs, deterministic=False)

                # 환경 스텝 (배치)
                next_obs, rewards, terminated, truncated, info = self.env.step(actions)
                dones = terminated | truncated

                # Transition 저장 (배치)
                self.agent.store_transition(obs, actions, rewards, next_obs, dones)

                # 환경별 통계 업데이트
                for env_id in range(num_envs):
                    episode_rewards[env_id] += rewards[env_id]
                    episode_lengths[env_id] += 1
                    total_steps += 1
                    self.agent.total_steps += 1

                    # 에피소드 종료
                    if dones[env_id]:
                        episode_num += 1
                        self.agent.episodes += 1

                        # 에피소드 통계 기록
                        self.episode_rewards.append(episode_rewards[env_id])
                        self.episode_lengths.append(episode_lengths[env_id])

                        # 로깅
                        self.logger.log_episode(
                            episode=episode_num,
                            episode_reward=episode_rewards[env_id],
                            episode_length=episode_lengths[env_id],
                            additional_metrics={
                                'env_id': env_id,
                                'best_score': info.get('best_score', [0] * num_envs)[env_id] if isinstance(info.get('best_score'), (list, np.ndarray)) else info.get('best_score', 0)
                            }
                        )

                        # 콘솔 출력
                        if episode_num % self.log_interval == 0:
                            mean_reward = np.mean(self.episode_rewards[-100:])
                            mean_length = np.mean(self.episode_lengths[-100:])
                            self.logger.print_progress(
                                step=total_steps,
                                total_steps=self.total_timesteps,
                                metrics={
                                    'episode': episode_num,
                                    'mean_reward': mean_reward,
                                    'mean_length': mean_length
                                }
                            )

                        # 환경별 통계 초기화
                        episode_rewards[env_id] = 0.0
                        episode_lengths[env_id] = 0

                # 다음 관찰로 이동
                obs = next_obs

                # Progress bar 업데이트 (num_envs만큼)
                pbar.update(num_envs)

                # 학습 (주기적으로)
                if total_steps % self.config.get('training', {}).get('update_frequency', 1) == 0:
                    update_info = self.agent.update()

                    # 학습 메트릭 로깅
                    if update_info and total_steps % self.log_interval == 0:
                        self.logger.log_training(total_steps, update_info.get('loss', 0), update_info)

                # 평가
                if total_steps > 0 and total_steps % self.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    mean_eval_reward = eval_metrics['mean_reward']

                    print(f"\nEvaluation at step {total_steps}:")
                    print(f"  Mean reward: {mean_eval_reward:.2f}")
                    print(f"  Mean length: {eval_metrics['mean_length']:.2f}")

                    # 최고 성능 모델 저장
                    if mean_eval_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_eval_reward
                        saved_name = self.save_checkpoint(suffix='best')
                        print(f"  New best model saved: {saved_name} (reward: {mean_eval_reward:.2f})")

                # 정기 체크포인트 저장
                if total_steps > 0 and total_steps % self.save_freq == 0:
                    saved_name = self.save_checkpoint(suffix=f'step{total_steps}')
                    print(f"Checkpoint saved: {saved_name}")

        # 학습 종료
        print("\nTraining completed!")
        saved_name = self.save_checkpoint(suffix='final')
        print(f"Final model saved: {saved_name}")
        self.logger.close()

    def evaluate(self, num_episodes: Optional[int] = None) -> Dict[str, float]:
        """
        에이전트 평가 (VectorEnv 지원)

        Args:
            num_episodes: 평가 에피소드 수 (None이면 설정값 사용)

        Returns:
            평가 메트릭
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        # 평가 모드로 전환
        self.agent.eval()

        # VectorEnv 감지
        num_envs = getattr(self.env, 'num_envs', 1)

        eval_rewards = []
        eval_lengths = []

        # 환경별 상태
        episode_rewards = [0.0] * num_envs
        episode_lengths = [0] * num_envs
        episodes_completed = [False] * num_envs

        obs, info = self.env.reset()

        while len(eval_rewards) < num_episodes:
            # 결정적 행동 선택 (배치)
            actions = self.agent.select_action(obs, deterministic=True)

            # 환경 스텝 (배치)
            obs, rewards, terminated, truncated, info = self.env.step(actions)
            dones = terminated | truncated

            # 환경별 처리
            for env_id in range(num_envs):
                if not episodes_completed[env_id]:
                    episode_rewards[env_id] += rewards[env_id]
                    episode_lengths[env_id] += 1

                    if dones[env_id]:
                        eval_rewards.append(episode_rewards[env_id])
                        eval_lengths.append(episode_lengths[env_id])

                        episode_rewards[env_id] = 0.0
                        episode_lengths[env_id] = 0

                        # 충분한 에피소드 수집 시 해당 환경 비활성화
                        if len(eval_rewards) >= num_episodes:
                            episodes_completed[env_id] = True

        # 학습 모드로 복귀
        self.agent.train()

        metrics = {
            'mean_reward': np.mean(eval_rewards[:num_episodes]),
            'std_reward': np.std(eval_rewards[:num_episodes]),
            'mean_length': np.mean(eval_lengths[:num_episodes]),
            'std_length': np.std(eval_lengths[:num_episodes])
        }

        return metrics

    def _generate_checkpoint_name(self, suffix: str) -> str:
        """
        동적으로 체크포인트 이름 생성

        Args:
            suffix: 'best', 'final', 'step_1000' 등

        Returns:
            생성된 파일명 (예: 'DQNAgent_1224_1430_best.pth')
        """
        # Agent 타입 추출
        agent_type = self.agent.__class__.__name__

        # 현재 시간 (MMDD_HHMM 형식)
        now = datetime.now()
        timestamp = now.strftime("%m%d_%H%M")

        # 파일명 생성: {agent_type}_{MMDD_HHMM}_{suffix}.pth
        filename = f"{agent_type}_{timestamp}_{suffix}.pth"

        return filename

    def save_checkpoint(self, filename: str = None, suffix: str = None) -> None:
        """
        체크포인트 저장

        Args:
            filename: 저장 파일명 (None이면 동적 생성)
            suffix: 동적 생성 시 사용할 suffix ('best', 'final' 등)
        """
        if filename is None:
            if suffix is None:
                suffix = 'checkpoint'
            filename = self._generate_checkpoint_name(suffix)

        path = self.save_path / filename
        self.agent.save(str(path))

        return filename  # 생성된 파일명 반환

    def load_checkpoint(self, filename: str) -> None:
        """
        체크포인트 로드

        Args:
            filename: 로드 파일명
        """
        path = self.save_path / filename
        self.agent.load(str(path))
