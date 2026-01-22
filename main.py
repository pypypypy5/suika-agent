"""
Suika Game RL 메인 실행 파일

이 파일은 학습 또는 평가를 시작합니다.
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import imageio

from envs import SuikaEnvWrapper
from agents import RandomAgent, SimpleAgent
from utils import setup_logger
from training import Trainer


def load_config(config_path: str) -> dict:
    """
    YAML 설정 파일 로드

    Args:
        config_path: 설정 파일 경로

    Returns:
        설정 딕셔너리
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, num_envs: int = None):
    """
    환경 생성 (항상 VectorEnv 반환)

    Args:
        config: 설정 딕셔너리
        num_envs: 환경 개수 (None이면 config.system.num_workers 사용)

    Returns:
        VectorEnv (SyncVectorEnv 또는 AsyncVectorEnv)
    """
    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

    env_config = config.get('env', {})
    system_config = config.get('system', {})

    if num_envs is None:
        num_envs = system_config.get('num_workers', 1)

    def make_env(rank):
        def _init():
            return SuikaEnvWrapper(
                headless=env_config.get('headless', True),
                port=env_config.get('port', 8923) + rank,  # 각 환경마다 고유 포트
                delay_before_img_capture=env_config.get('delay_before_img_capture', 0.5),
                observation_type=env_config.get('observation_type', 'image'),
                reward_scale=env_config.get('reward_scale', 1.0),
                normalize_obs=env_config.get('normalize_obs', True),
                use_mock=env_config.get('use_mock', False),
                fast_mode=env_config.get('fast_mode', True)
            )
        return _init

    envs = [make_env(i) for i in range(num_envs)]

    # num_envs=1이면 SyncVectorEnv (오버헤드 최소)
    # num_envs>1이면 AsyncVectorEnv (병렬 처리)
    if num_envs == 1:
        vec_env = SyncVectorEnv(envs)
        print(f"Created SyncVectorEnv with {num_envs} environment")
    else:
        vec_env = AsyncVectorEnv(envs)
        print(f"Created AsyncVectorEnv with {num_envs} environments")

    print(f"Observation space: {vec_env.single_observation_space}")
    print(f"Action space: {vec_env.single_action_space}")

    return vec_env


def create_agent(env, config: dict):
    """
    에이전트 생성

    Args:
        env: VectorEnv (create_env에서 항상 VectorEnv 반환)
        config: 설정 딕셔너리

    Returns:
        에이전트
    """
    agent_config = config.get('agent', {})
    agent_type = agent_config.get('type', 'random')

    # VectorEnv의 single_observation_space와 single_action_space 사용
    # (환경이 1개든 여러 개든 동일한 인터페이스)
    obs_space = env.single_observation_space
    act_space = env.single_action_space

    if agent_type == 'random':
        # 랜덤 에이전트 (베이스라인)
        agent = RandomAgent(
            observation_space=obs_space,
            action_space=act_space,
            config=agent_config
        )
        print("Using Random Agent (baseline)")
    elif agent_type == 'simple':
        # 간단한 Policy Gradient 에이전트
        agent = SimpleAgent(
            observation_space=obs_space,
            action_space=act_space,
            config=agent_config
        )
        print("Using Simple Policy Gradient Agent")
    else:
        # 여기에 다른 에이전트 타입 추가
        # 예: DQN, PPO, SAC 등
        raise NotImplementedError(
            f"Agent type '{agent_type}' not implemented yet. "
            f"Please implement your agent in agents/ directory and add it here."
        )

    return agent


def train(config: dict, experiment_name: str = None) -> None:
    """
    학습 실행

    Args:
        config: 설정 딕셔너리
        experiment_name: 실험 이름
    """
    print("=" * 60)
    print("SUIKA GAME REINFORCEMENT LEARNING - TRAINING")
    print("=" * 60)

    # 환경 생성
    env = create_env(config)

    # 에이전트 생성
    agent = create_agent(env, config)

    # 로거 설정
    logger = setup_logger(config, experiment_name)

    # 트레이너 생성 및 학습
    trainer = Trainer(
        agent=agent,
        env=env,
        logger=logger,
        config=config
    )

    trainer.train()

    env.close()


def evaluate(config: dict, checkpoint_path: str) -> None:
    """
    학습된 모델 평가

    Args:
        config: 설정 딕셔너리
        checkpoint_path: 체크포인트 경로
    """
    print("=" * 60)
    print("SUIKA GAME REINFORCEMENT LEARNING - EVALUATION")
    print("=" * 60)

    # 환경 생성 (기존 env 코드는 건드리지 않음)
    env = create_env(config)

    # 비디오 저장 폴더 생성
    video_folder = Path("experiments/videos")
    video_folder.mkdir(parents=True, exist_ok=True)
    print(f"Videos will be saved to: {video_folder}")

    # 에이전트 생성
    agent = create_agent(env, config)

    # 체크포인트 로드
    print(f"Loading checkpoint from {checkpoint_path}")
    agent.load(checkpoint_path)

    # 평가
    agent.eval()
    num_eval_episodes = config.get('training', {}).get('eval_episodes', 10)

    print(f"\nEvaluating for {num_eval_episodes} episodes...")

    episode_rewards = []
    for episode in range(num_eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # 프레임 저장을 위한 리스트 (수동 비디오 저장)
        frames = []

        while not done:
            # 현재 프레임 저장 (observation에서 'image' 추출)
            if isinstance(obs, dict) and 'image' in obs:
                frame = obs['image']
                # 정규화된 이미지를 uint8로 변환
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (frame * 255).astype(np.uint8)
                # RGBA를 RGB로 변환 (alpha 채널 제거)
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                frames.append(frame)

            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # 마지막 프레임도 저장
        if isinstance(obs, dict) and 'image' in obs:
            frame = obs['image']
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            frames.append(frame)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        # 비디오 저장 (각 에피소드마다)
        if frames:
            video_path = video_folder / f"eval-episode-{episode}.mp4"
            try:
                imageio.mimsave(str(video_path), frames, fps=30)
                print(f"  Video saved: {video_path}")
            except Exception as e:
                print(f"  Warning: Failed to save video: {e}")

    # 통계 출력
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"\nVideos saved to: {video_folder}")

    env.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Suika Game Reinforcement Learning"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval'],
        default='train',
        help='실행 모드: train (학습) 또는 eval (평가)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='체크포인트 경로 (평가 모드에서 필요)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='실험 이름 (학습 모드)'
    )

    args = parser.parse_args()

    # 설정 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))
    print(f"Loaded config from {config_path}")

    # 모드에 따라 실행
    if args.mode == 'train':
        train(config, args.experiment_name)
    elif args.mode == 'eval':
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluation mode")
        evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
