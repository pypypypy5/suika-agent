"""
Test HTTP-based Suika environment
"""

import sys
import time
import numpy as np

# Add suika_rl to path
sys.path.insert(0, 'suika_rl')

from suika_env.suika_http_env import SuikaBrowserEnv


def test_basic():
    """Test basic environment functionality"""
    print("=" * 60)
    print("Test 1: Basic Environment Creation and Reset")
    print("=" * 60)

    env = SuikaBrowserEnv(port=8924)
    print(f"[OK] Environment created")

    obs, info = env.reset()
    print(f"[OK] Environment reset")
    print(f"  - Observation keys: {obs.keys()}")
    print(f"  - Image shape: {obs['image'].shape}")
    print(f"  - Score shape: {obs['score'].shape}")
    print(f"  - Initial score: {obs['score'][0]}")

    assert obs['image'].shape == (128, 128, 3), "Image shape mismatch"
    assert obs['score'].shape == (1,), "Score shape mismatch"
    assert obs['score'][0] == 0, "Initial score should be 0"

    print("[OK] All assertions passed\n")
    env.close()


def test_step():
    """Test environment step function"""
    print("=" * 60)
    print("Test 2: Environment Step")
    print("=" * 60)

    env = SuikaBrowserEnv(port=8924)
    obs, info = env.reset()

    # Take 5 steps
    for i in range(5):
        action = np.array([np.random.rand()], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  - Action: {action[0]:.3f}")
        print(f"  - Score: {obs['score'][0]}")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Image shape: {obs['image'].shape}")

        if terminated:
            print("  - Game over!")
            break

    print("[OK] Step function works\n")
    env.close()


def test_performance():
    """Test environment performance (steps per second)"""
    print("=" * 60)
    print("Test 3: Performance Benchmark")
    print("=" * 60)

    env = SuikaBrowserEnv(port=8924)
    obs, info = env.reset()

    num_steps = 20
    start_time = time.time()

    for i in range(num_steps):
        action = np.array([np.random.rand()], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            obs, info = env.reset()

    elapsed = time.time() - start_time
    steps_per_sec = num_steps / elapsed

    print(f"Completed {num_steps} steps in {elapsed:.2f} seconds")
    print(f"Performance: {steps_per_sec:.2f} steps/sec")
    print(f"Average time per step: {elapsed/num_steps*1000:.1f} ms")

    if steps_per_sec > 10:
        print("[OK] Performance is EXCELLENT (>10 steps/sec)")
    elif steps_per_sec > 5:
        print("[OK] Performance is GOOD (>5 steps/sec)")
    elif steps_per_sec > 1:
        print("[WARN] Performance is OK (>1 steps/sec)")
    else:
        print("[FAIL] Performance is POOR (<1 steps/sec)")

    print()
    env.close()


def test_parallel():
    """Test multiple parallel environments"""
    print("=" * 60)
    print("Test 4: Parallel Environments")
    print("=" * 60)

    num_envs = 4
    envs = [SuikaBrowserEnv(port=8924) for _ in range(num_envs)]
    print(f"[OK] Created {num_envs} parallel environments")

    # Reset all
    for i, env in enumerate(envs):
        obs, info = env.reset()
        print(f"  - Env {i}: reset OK, score={obs['score'][0]}")

    # Take steps in all
    for i, env in enumerate(envs):
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  - Env {i}: step OK, score={obs['score'][0]}")

    print("[OK] All parallel environments work\n")

    for env in envs:
        env.close()


if __name__ == "__main__":
    try:
        test_basic()
        test_step()
        test_performance()
        test_parallel()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
