"""
사용 예제

환경과 에이전트의 기본적인 사용법을 보여주는 예제
"""

import numpy as np
from envs import SuikaEnvWrapper
from agents import RandomAgent


def example_environment_interaction():
    """
    환경과의 기본 상호작용 예제
    """
    print("=" * 60)
    print("환경 상호작용 예제")
    print("=" * 60)

    # 1. 환경 생성
    env = SuikaEnvWrapper(
        observation_type="image",
        reward_scale=1.0,
        normalize_obs=True
    )

    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # 2. 에피소드 실행
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        # 환경 리셋
        obs, info = env.reset()
        print(f"Initial observation shape: {obs['image'].shape if isinstance(obs, dict) else obs.shape}")

        episode_reward = 0
        episode_length = 0

        # 에피소드 진행
        for step in range(100):  # 최대 100 스텝
            # 랜덤 행동 선택
            action = env.action_space.sample()

            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # 에피소드 종료
            if terminated or truncated:
                break

        print(f"Episode finished after {episode_length} steps")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Episode score: {info.get('episode_score', 0):.2f}")

    env.close()


def example_agent_usage():
    """
    에이전트 사용 예제
    """
    print("\n" + "=" * 60)
    print("에이전트 사용 예제")
    print("=" * 60)

    # 1. 환경 생성
    env = SuikaEnvWrapper()

    # 2. 에이전트 생성
    agent = RandomAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )

    print(f"\nAgent: {agent.__class__.__name__}")

    # 3. 에이전트로 에피소드 실행
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        obs, info = env.reset()
        episode_reward = 0

        for step in range(100):
            # 에이전트가 행동 선택
            action = agent.select_action(obs)

            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        print(f"Total reward: {episode_reward:.2f}")

    env.close()


def example_custom_reward():
    """
    커스텀 보상 함수 예제
    """
    print("\n" + "=" * 60)
    print("커스텀 보상 함수 예제")
    print("=" * 60)

    # 보상 스케일 조정
    env = SuikaEnvWrapper(reward_scale=0.01)

    print("\n보상이 0.01배로 스케일링됩니다.")

    obs, info = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Original reward: {info.get('original_reward', 0):.2f}, "
              f"Processed reward: {reward:.4f}")

        if terminated or truncated:
            break

    env.close()


def example_statistics():
    """
    통계 정보 활용 예제
    """
    print("\n" + "=" * 60)
    print("통계 정보 활용 예제")
    print("=" * 60)

    env = SuikaEnvWrapper()

    # 여러 에피소드 실행
    for episode in range(5):
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # 에피소드 통계 확인
        stats = env.get_episode_statistics()
        print(f"\nEpisode {episode + 1} Statistics:")
        print(f"  Score: {stats['episode_score']:.2f}")
        print(f"  Steps: {stats['episode_steps']}")
        print(f"  Average reward: {stats['average_reward']:.4f}")
        print(f"  Best score so far: {stats['best_score']:.2f}")

    env.close()


if __name__ == "__main__":
    # 모든 예제 실행
    example_environment_interaction()
    example_agent_usage()
    example_custom_reward()
    example_statistics()

    print("\n" + "=" * 60)
    print("모든 예제 완료!")
    print("=" * 60)
    print("\n이제 agents/ 디렉토리에 자신만의 에이전트를 구현할 수 있습니다.")
    print("BaseAgent 또는 RLAgent 클래스를 상속받아 구현하세요.")
    print("\n학습을 시작하려면:")
    print("  python main.py --mode train --config config/default.yaml")
