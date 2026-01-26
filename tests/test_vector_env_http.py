"""
VectorEnv with HTTP backend 테스트
main.py와 동일한 방식으로 환경 생성 및 테스트
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from envs import SuikaEnvWrapper


base_port = 8924


def test_single_env():
    """단일 환경 테스트 (SyncVectorEnv)"""
    print("=" * 70)
    print("Test 1: Single Environment (SyncVectorEnv)")
    print("=" * 70)

    def make_env():
        return SuikaEnvWrapper(
            headless=True,
            port=base_port,
            observation_type='image',
            normalize_obs=True,
            use_mock=False,
            fast_mode=True,
            auto_start_server=True
        )

    vec_env = SyncVectorEnv([make_env])
    print(f"[OK] SyncVectorEnv created")
    print(f"  - Num envs: {vec_env.num_envs}")
    print(f"  - Single observation space: {vec_env.single_observation_space}")
    print(f"  - Single action space: {vec_env.single_action_space}")

    # Reset (VectorEnv returns tuple)
    obs, infos = vec_env.reset()
    print(f"[OK] Environment reset")
    print(f"  - Observation shape: {obs['image'].shape}")
    print(f"  - Score shape: {obs['score'].shape}")

    # Take steps
    for i in range(5):
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        print(f"Step {i+1}: reward={rewards[0]:.1f}, done={terminated[0] or truncated[0]}")

    vec_env.close()
    print("[OK] Single environment test passed\n")


def test_multi_env():
    """멀티 환경 테스트 (AsyncVectorEnv) - main.py와 동일"""
    print("=" * 70)
    print("Test 2: Multi Environment (AsyncVectorEnv, num_workers=4)")
    print("=" * 70)

    num_envs = 4

    def make_env(rank):
        def _init():
            return SuikaEnvWrapper(
                headless=True,
                port=(base_port + rank),  # 모든 워커가 같은 포트 공유
                observation_type='image',
                normalize_obs=True,
                use_mock=False,
                fast_mode=True,
                auto_start_server=True
            )
        return _init

    envs = [make_env(i) for i in range(num_envs)]
    vec_env = AsyncVectorEnv(envs)

    print(f"[OK] AsyncVectorEnv created")
    print(f"  - Num envs: {vec_env.num_envs}")
    print(f"  - Single observation space: {vec_env.single_observation_space}")
    print(f"  - Single action space: {vec_env.single_action_space}")

    # Reset (VectorEnv returns tuple)
    obs, infos = vec_env.reset()
    print(f"[OK] All environments reset")
    print(f"  - Observation shape: {obs['image'].shape}")
    print(f"  - Score shape: {obs['score'].shape}")

    # Take steps
    total_steps = 10
    for i in range(total_steps):
        actions = vec_env.action_space.sample()
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        done_count = sum(terminated) + sum(truncated)
        avg_reward = np.mean(rewards)
        print(f"Step {i+1}: avg_reward={avg_reward:.1f}, done_count={done_count}/{num_envs}")

    vec_env.close()
    print("[OK] Multi environment test passed\n")


def test_from_config():
    """config 파일을 사용한 환경 생성 (main.py 방식)"""
    print("=" * 70)
    print("Test 3: Create Environment from Config (main.py style)")
    print("=" * 70)

    import yaml

    # Load config
    with open('config/default.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"[OK] Config loaded")
    print(f"  - Port: {config['env']['port']}")
    print(f"  - Fast mode: {config['env']['fast_mode']}")
    print(f"  - Num workers: {config['system']['num_workers']}")

    # Create env (main.py 방식)
    env_config = config.get('env', {})
    system_config = config.get('system', {})
    num_envs = system_config.get('num_workers', 4)
    auto_start_per_env = env_config.get('auto_start_server_per_env', True)
    base = env_config.get('port_base', env_config.get('port', 8924))

    def make_env(rank):
        def _init():
            return SuikaEnvWrapper(
                headless=env_config.get('headless', True),
                port=env_config.get('port', 8924),  # 모든 워커가 같은 포트 공유
                observation_type=env_config.get('observation_type', 'image'),
                reward_scale=env_config.get('reward_scale', 1.0),
                normalize_obs=env_config.get('normalize_obs', True),
                use_mock=env_config.get('use_mock', False),
                fast_mode=env_config.get('fast_mode', True),
                auto_start_server=auto_start_per_env
            )
        return _init

    envs = [make_env(i) for i in range(num_envs)]

    if num_envs == 1:
        vec_env = SyncVectorEnv(envs)
        print(f"[OK] Created SyncVectorEnv with {num_envs} environment")
    else:
        vec_env = AsyncVectorEnv(envs)
        print(f"[OK] Created AsyncVectorEnv with {num_envs} environments")

    # Quick test (VectorEnv returns tuple)
    obs, infos = vec_env.reset()
    print(f"[OK] Environment reset")

    actions = vec_env.action_space.sample()
    obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    print(f"[OK] Step executed")
    print(f"  - Rewards: {rewards}")
    print(f"  - Terminated: {terminated}")

    vec_env.close()
    print("[OK] Config-based environment test passed\n")


if __name__ == "__main__":

    try:
        test_single_env()
        test_multi_env()
        test_from_config()

        print("=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        print("\nReady to run main.py for full training.")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
