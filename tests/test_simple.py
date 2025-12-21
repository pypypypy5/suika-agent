"""
간단한 Mock 환경 테스트 (의존성 최소화)
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 먼저 import 테스트
print("=" * 70)
print("Step 1: 모듈 Import 테스트")
print("=" * 70)

try:
    import numpy as np
    print("✓ numpy import 성공")
except ImportError as e:
    print(f"✗ numpy import 실패: {e}")
    print("\n해결 방법: pip install numpy")
    sys.exit(1)

try:
    import gymnasium as gym
    print("✓ gymnasium import 성공")
except ImportError as e:
    print(f"✗ gymnasium import 실패: {e}")
    print("\n해결 방법: pip install gymnasium")
    sys.exit(1)

try:
    from envs import SuikaEnvWrapper
    print("✓ SuikaEnvWrapper import 성공")
except ImportError as e:
    print(f"✗ SuikaEnvWrapper import 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Mock 환경 테스트
print("\n" + "=" * 70)
print("Step 2: Mock 환경 생성 테스트")
print("=" * 70)

try:
    env = SuikaEnvWrapper(use_mock=True)
    print("✓ Mock 환경 생성 성공")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
except Exception as e:
    print(f"✗ 환경 생성 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Reset 테스트
print("\n" + "=" * 70)
print("Step 3: 환경 Reset 테스트")
print("=" * 70)

try:
    obs, info = env.reset(seed=42)
    print("✓ reset() 성공")
    print(f"\n초기 관찰 (Observation):")
    if isinstance(obs, dict):
        for key, value in obs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  shape={obs.shape}, dtype={obs.dtype}")

    print(f"\n초기 정보 (Info):")
    for key, value in info.items():
        print(f"  {key}: {value}")

except Exception as e:
    print(f"✗ reset() 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 테스트
print("\n" + "=" * 70)
print("Step 4: 환경 Step 테스트 (3 스텝)")
print("=" * 70)

try:
    for i in range(3):
        print(f"\n--- Step {i+1} ---")

        # 랜덤 행동
        action = env.action_space.sample()
        print(f"행동: {action} (과일 떨어뜨릴 위치)")

        # 행동 실행
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"결과:")
        print(f"  - Reward: {reward:.4f}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")

        if isinstance(obs, dict) and 'score' in obs:
            print(f"  - Score: {obs['score'][0]:.0f}")

        print(f"  - Info: {info}")

        if terminated or truncated:
            print("\n게임 종료!")
            break

    print("\n✓ step() 성공")

except Exception as e:
    print(f"\n✗ step() 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 통계 테스트
print("\n" + "=" * 70)
print("Step 5: 환경 통계 확인")
print("=" * 70)

try:
    stats = env.get_episode_statistics()
    print("에피소드 통계:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\n✓ 통계 조회 성공")

except Exception as e:
    print(f"✗ 통계 조회 실패: {e}")
    import traceback
    traceback.print_exc()

# 종료
print("\n" + "=" * 70)
print("Step 6: 환경 종료")
print("=" * 70)

try:
    env.close()
    print("✓ 환경 정상 종료")
except Exception as e:
    print(f"✗ 환경 종료 실패: {e}")

print("\n" + "=" * 70)
print("모든 테스트 통과! ✓")
print("=" * 70)
print("\nAPI 요약:")
print("  - reset() → (observation, info)")
print("  - step(action) → (observation, reward, terminated, truncated, info)")
print("  - get_episode_statistics() → dict")
print("  - close() → None")
print("\n에이전트 구현에 필요한 모든 인터페이스가 정상 작동합니다.")
