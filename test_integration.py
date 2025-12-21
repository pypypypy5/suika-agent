"""
통합 테스트 스크립트

실제 Suika 환경과의 통합을 테스트합니다.
"""

import sys
import time
from envs import SuikaEnvWrapper


def test_mock_environment():
    """Mock 환경 테스트"""
    print("=" * 60)
    print("TEST 1: Mock 환경 테스트")
    print("=" * 60)

    try:
        env = SuikaEnvWrapper(use_mock=True)
        print("✓ Mock 환경 생성 성공")

        obs, info = env.reset()
        print(f"✓ 환경 리셋 성공")
        print(f"  - Observation keys: {obs.keys() if isinstance(obs, dict) else 'not dict'}")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")

        # 몇 스텝 실행
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.2f}, done={terminated or truncated}")

            if terminated or truncated:
                break

        env.close()
        print("✓ Mock 환경 테스트 완료\n")
        return True

    except Exception as e:
        print(f"✗ Mock 환경 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_real_environment():
    """실제 Suika 환경 테스트"""
    print("=" * 60)
    print("TEST 2: 실제 Suika 환경 테스트")
    print("=" * 60)

    # Chrome/Chromium이 설치되어 있는지 확인
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Chrome 드라이버 테스트
        print("Chrome 드라이버 테스트 중...")
        driver = webdriver.Chrome(options=options)
        driver.quit()
        print("✓ Chrome 드라이버 사용 가능")

    except Exception as e:
        print(f"✗ Chrome 드라이버 사용 불가: {e}")
        print("  Selenium과 Chrome/Chromium 설치가 필요합니다.")
        print("  설치 방법:")
        print("    - Ubuntu/Debian: sudo apt-get install chromium-browser chromium-chromedriver")
        print("    - macOS: brew install chromedriver")
        print("    - Windows: https://chromedriver.chromium.org/ 에서 다운로드")
        return False

    # 실제 환경 테스트
    try:
        print("\n실제 Suika 환경 생성 중...")
        env = SuikaEnvWrapper(
            headless=True,
            port=8924,  # 다른 포트 사용
            delay_before_img_capture=0.5,
            use_mock=False
        )
        print("✓ 실제 환경 생성 성공")

        print("환경 리셋 중...")
        obs, info = env.reset()
        print(f"✓ 환경 리셋 성공")
        print(f"  - Observation keys: {obs.keys()}")
        print(f"  - Image shape: {obs['image'].shape}")
        print(f"  - Score: {obs['score']}")

        # 몇 스텝 실행
        print("\n몇 스텝 실행 중...")
        for i in range(3):
            action = env.action_space.sample()
            print(f"  Step {i+1}: action={action}")

            obs, reward, terminated, truncated, info = env.step(action)
            print(f"    → reward={reward:.2f}, score={obs['score'][0]:.0f}, done={terminated or truncated}")

            if terminated or truncated:
                print("    게임 종료!")
                break

            time.sleep(0.5)  # 관찰을 위한 대기

        env.close()
        print("\n✓ 실제 환경 테스트 완료\n")
        return True

    except Exception as e:
        print(f"\n✗ 실제 환경 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper_features():
    """래퍼 기능 테스트"""
    print("=" * 60)
    print("TEST 3: 래퍼 기능 테스트")
    print("=" * 60)

    try:
        # 보상 스케일링 테스트
        env = SuikaEnvWrapper(use_mock=True, reward_scale=0.01)
        print("✓ 보상 스케일링 래퍼 생성")

        obs, info = env.reset()
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Original: {info.get('original_reward', 0):.2f}, "
                  f"Scaled: {reward:.4f}")
            if terminated or truncated:
                break

        # 통계 확인
        stats = env.get_episode_statistics()
        print(f"\n✓ 에피소드 통계:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        env.close()
        print("\n✓ 래퍼 기능 테스트 완료\n")
        return True

    except Exception as e:
        print(f"\n✗ 래퍼 기능 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("SUIKA RL 통합 테스트")
    print("=" * 60 + "\n")

    results = []

    # Test 1: Mock 환경
    results.append(("Mock 환경", test_mock_environment()))

    # Test 2: 실제 환경 (선택적)
    print("실제 Suika 환경을 테스트하시겠습니까?")
    print("(Chrome/Chromium과 Chromedriver가 필요합니다)")
    response = input("y/N: ").strip().lower()

    if response == 'y':
        results.append(("실제 Suika 환경", test_real_environment()))
    else:
        print("실제 환경 테스트 건너뜀\n")

    # Test 3: 래퍼 기능
    results.append(("래퍼 기능", test_wrapper_features()))

    # 결과 요약
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ 통과" if passed else "✗ 실패"
        print(f"{test_name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n총 {total}개 테스트 중 {passed}개 통과")

    if passed == total:
        print("\n모든 테스트 통과! 🎉")
    else:
        print(f"\n{total - passed}개 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
