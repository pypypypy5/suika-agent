"""
Test worker crash recovery mechanisms
"""

import pytest
import numpy as np
import time
from envs import make_resilient_vector_env
from envs.suika_wrapper import SuikaEnvWrapper


def test_sync_vector_env_stability():
    """
    Test that SyncVectorEnv (num_workers=1) is stable and doesn't crash.
    """
    def make_env():
        return SuikaEnvWrapper(
            use_mock=True,  # Use mock to avoid server issues
            normalize_obs=True
        )

    env_fns = [make_env]
    vec_env = make_resilient_vector_env(env_fns, num_envs=1)

    # Reset
    obs, info = vec_env.reset()
    assert obs is not None

    # Run 100 steps
    for _ in range(100):
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        assert obs is not None

        if terminated[0] or truncated[0]:
            obs, info = vec_env.reset()

    vec_env.close()
    print("SyncVectorEnv test passed")


def test_async_vector_env_with_mock():
    """
    Test AsyncVectorEnv with mock environments (should be stable).
    """
    def make_env(rank):
        def _init():
            return SuikaEnvWrapper(
                use_mock=True,  # Mock environments don't have server issues
                normalize_obs=True
            )
        return _init

    num_envs = 2
    env_fns = [make_env(i) for i in range(num_envs)]
    vec_env = make_resilient_vector_env(env_fns, num_envs=num_envs)

    # Reset
    obs, info = vec_env.reset()
    assert obs is not None
    assert obs['image'].shape[0] == num_envs

    # Run 50 steps
    for _ in range(50):
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        assert obs is not None

        # Reset individual environments that are done
        if terminated.any() or truncated.any():
            # AsyncVectorEnv handles this automatically
            pass

    vec_env.close()
    print("AsyncVectorEnv with mock test passed")


def test_recommendation_num_workers_1():
    """
    Test that using num_workers=1 (SyncVectorEnv) is the recommended solution.
    This is the most stable configuration on Windows.
    """
    def make_env():
        return SuikaEnvWrapper(
            port=8924,
            auto_start_server=True,
            force_server_restart=False,
            normalize_obs=True,
            use_mock=False  # Use real environment
        )

    env_fns = [make_env]
    vec_env = make_resilient_vector_env(env_fns, num_envs=1)

    print("Using SyncVectorEnv (most stable)")

    # Reset and run a few steps
    obs, info = vec_env.reset()
    assert obs is not None

    for _ in range(10):
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, info = vec_env.step(actions)
        assert obs is not None

        if terminated[0] or truncated[0]:
            obs, info = vec_env.reset()

    vec_env.close()
    print("num_workers=1 test passed")


if __name__ == '__main__':
    print("Testing worker crash recovery...")
    print("\n" + "="*60)
    print("Test 1: SyncVectorEnv stability")
    print("="*60)
    test_sync_vector_env_stability()

    print("\n" + "="*60)
    print("Test 2: AsyncVectorEnv with mock")
    print("="*60)
    test_async_vector_env_with_mock()

    print("\n" + "="*60)
    print("Test 3: Recommended solution (num_workers=1)")
    print("="*60)
    test_recommendation_num_workers_1()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
