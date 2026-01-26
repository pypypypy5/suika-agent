"""
Test actual training pipeline with HTTP environment
"""

import sys
import yaml
import numpy as np

# Test imports
from envs.suika_wrapper import SuikaEnvWrapper
from agents.DQN_agent import DQNAgent
from training.trainer import Trainer


def test_wrapper():
    """Test that SuikaEnvWrapper works with HTTP backend"""
    print("=" * 60)
    print("Test 1: SuikaEnvWrapper Integration")
    print("=" * 60)

    env = SuikaEnvWrapper(
        headless=True,
        port=8924,
        observation_type="image",
        fast_mode=True
    )

    print(f"[OK] Wrapper created")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"[OK] Environment reset")
    print(f"  - Observation keys: {obs.keys()}")

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"[OK] Step executed")
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}\n")

    env.close()


def test_agent_creation():
    """Test that DQN agent can be created"""
    print("=" * 60)
    print("Test 2: DQN Agent Creation")
    print("=" * 60)

    # Load config
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = SuikaEnvWrapper(
        headless=True,
        port=8924,
        observation_type="image",
        fast_mode=True
    )

    # Create agent
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config
    )

    print(f"[OK] DQN Agent created")
    print(f"  - Agent type: {type(agent).__name__}\n")

    env.close()


def test_mini_training():
    """Test mini training loop (10 steps)"""
    print("=" * 60)
    print("Test 3: Mini Training Loop (10 steps)")
    print("=" * 60)

    # Load config
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override to very small values
    config['training']['total_timesteps'] = 10

    # Create environment
    env = SuikaEnvWrapper(
        headless=True,
        port=8924,
        observation_type="image",
        fast_mode=True,
        max_episode_steps=10
    )

    # Create agent
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config=config
    )

    print("[OK] Starting mini training...")

    # Reset
    obs, info = env.reset()

    for step in range(10):
        # Select action
        action = agent.select_action(obs)

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step+1}: reward={reward:.1f}, done={terminated}")

        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    print("[OK] Mini training completed (transition storage skipped)\n")

    env.close()


if __name__ == "__main__":
    try:
        test_wrapper()
        test_agent_creation()
        test_mini_training()

        print("=" * 60)
        print("ALL TRAINING TESTS PASSED!")
        print("=" * 60)
        print("\nReady to run full training:")
        print("  python main.py --config config/default.yaml")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
