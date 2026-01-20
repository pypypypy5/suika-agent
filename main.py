"""
Suika Game RL 메인 실행 파일

이 파일은 학습 또는 평가를 시작합니다.
"""

import argparse
import yaml
from pathlib import Path

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


def create_env(config: dict) -> SuikaEnvWrapper:
    """
    환경 생성

    Args:
        config: 설정 딕셔너리

    Returns:
        Suika 환경
    """
    env_config = config.get('env', {})

    env = SuikaEnvWrapper(
        headless=env_config.get('headless', True),
        port=env_config.get('port', 8923),
        delay_before_img_capture=env_config.get('delay_before_img_capture', 0.5),
        observation_type=env_config.get('observation_type', 'image'),
        reward_scale=env_config.get('reward_scale', 1.0),
        normalize_obs=env_config.get('normalize_obs', True),
        use_mock=env_config.get('use_mock', False)
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    return env


def create_agent(env: SuikaEnvWrapper, config: dict):
    """
    에이전트 생성

    Args:
        env: 환경
        config: 설정 딕셔너리

    Returns:
        에이전트
    """
    agent_config = config.get('agent', {})
    agent_type = agent_config.get('type', 'random')

    if agent_type == 'random':
        # 랜덤 에이전트 (베이스라인)
        agent = RandomAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=agent_config
        )
        print("Using Random Agent (baseline)")
    elif agent_type == 'simple':
        # 간단한 Policy Gradient 에이전트
        agent = SimpleAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
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

    # 환경 생성
    env = create_env(config)

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

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # 통계 출력
    import numpy as np
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

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
