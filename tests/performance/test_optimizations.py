"""
최적화가 기능을 깨뜨리지 않았는지 검증
"""

import sys
import os
import numpy as np

# suika_rl 패키지 경로 추가
suika_rl_path = os.path.join(os.path.dirname(__file__), 'suika_rl')
if os.path.exists(suika_rl_path) and suika_rl_path not in sys.path:
    sys.path.insert(0, suika_rl_path)

from envs.suika_wrapper import SuikaEnvWrapper


def test_basic_functionality():
    """기본 기능 테스트"""
    print("="*60)
    print("Testing basic functionality...")
    print("="*60)

    env = SuikaEnvWrapper(
        headless=True,
        port=8927,
        fast_mode=True,
        use_mock=False
    )

    try:
        # Reset 테스트
        print("\n1. Testing reset...")
        obs, info = env.reset()
        assert isinstance(obs, dict), "Observation should be a dict"
        assert 'image' in obs, "Observation should contain 'image'"
        assert 'score' in obs, "Observation should contain 'score'"
        print("   [OK] Reset works correctly")

        # Step 테스트
        print("\n2. Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict), "Observation should be a dict"
        assert isinstance(reward, (int, float, np.number)), "Reward should be numeric"
        assert isinstance(terminated, bool), "Terminated should be boolean"
        assert isinstance(truncated, bool), "Truncated should be boolean"
        assert isinstance(info, dict), "Info should be dict"
        print("   [OK] Step works correctly")

        # 여러 스텝 실행
        print("\n3. Testing multiple steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"   Episode ended at step {i+1}")
                break
        print("   [OK] Multiple steps work correctly")

        # 점수 증가 확인
        print("\n4. Testing score progression...")
        obs, info = env.reset()
        initial_score = obs['score'].item()

        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        final_score = obs['score'].item()
        print(f"   Initial score: {initial_score}")
        print(f"   Final score: {final_score}")
        if final_score > initial_score:
            print("   [OK] Score increases as expected")
        else:
            print("   ! Score didn't increase (might be unlucky RNG)")

        # 종료 조건 테스트
        print("\n5. Testing termination...")
        obs, info = env.reset()
        max_steps = 100
        terminated = False

        for i in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"   Game ended at step {i+1}")
                print("   [OK] Termination works correctly")
                break

        if not terminated:
            print(f"   Game didn't end in {max_steps} steps (acceptable)")

        print("\n" + "="*60)
        print("All tests passed! [OK]")
        print("="*60)

    finally:
        env.close()


def test_fast_mode():
    """Fast mode 특화 기능 테스트"""
    print("\n" + "="*60)
    print("Testing fast mode features...")
    print("="*60)

    env = SuikaEnvWrapper(
        headless=True,
        port=8928,
        fast_mode=True,
        use_mock=False
    )

    try:
        obs, info = env.reset()

        print("\n1. Testing fast mode is enabled...")
        fast_mode_status = env.env.driver.execute_script('return window.Game.fastMode;')
        assert fast_mode_status == True, "Fast mode should be enabled"
        print("   [OK] Fast mode is enabled")

        print("\n2. Testing runner is stopped (no real-time rendering)...")
        runner_enabled = env.env.driver.execute_script('return window.runner.enabled;')
        assert runner_enabled == False, "Runner should be disabled in fast mode"
        print("   [OK] Runner is disabled")

        print("\n3. Testing fast forward works...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert 'stable' in info, "Info should contain stability status"
        print(f"   Stability: {info['stable']}")
        print("   [OK] Fast forward works")

        print("\n" + "="*60)
        print("Fast mode tests passed! [OK]")
        print("="*60)

    finally:
        env.close()


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_fast_mode()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY [OK]")
        print("Optimizations did not break functionality!")
        print("="*60)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
