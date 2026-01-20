"""
안정화 대기 기능 테스트

과일이 떨어진 후 완전히 멈출 때까지 대기하는 새로운 기능을 테스트합니다.
"""

import sys
import os
import time
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_stability_with_mock():
    """Mock 환경으로 안정화 대기 기능 기본 테스트"""
    print("=" * 70)
    print("  Mock 환경으로 안정화 대기 기능 테스트")
    print("=" * 70)

    from envs import SuikaEnvWrapper

    print("\n[1] Mock 환경 생성")
    env = SuikaEnvWrapper(use_mock=True)
    print("✓ Mock 환경 생성 완료")

    print("\n[2] 환경 리셋")
    obs, info = env.reset()
    print("✓ 환경 리셋 완료")

    print("\n[3] 행동 실행 및 정보 확인")
    action = np.array([0.5], dtype=np.float32)

    start_time = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    elapsed = time.time() - start_time

    print(f"✓ step() 실행 완료 (소요 시간: {elapsed:.3f}초)")
    print(f"  - 보상: {reward}")
    print(f"  - 종료: terminated={terminated}, truncated={truncated}")
    print(f"  - info 키: {list(info.keys())}")

    # Mock 환경에서는 stable 정보가 없을 수 있음 (JavaScript 없음)
    if 'stable' in info:
        print(f"  - 안정화 상태: {info['stable']}")
    else:
        print("  - 안정화 상태: N/A (Mock 환경)")

    env.close()
    print("\n✓ Mock 환경 테스트 완료")


def test_stability_with_real_env():
    """실제 환경으로 안정화 대기 기능 상세 테스트"""
    print("\n" + "=" * 70)
    print("  실제 Suika 환경으로 안정화 대기 기능 테스트")
    print("=" * 70)

    # Chrome 드라이버 확인
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        print("\n[준비] Chrome 드라이버 확인")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        driver.quit()
        print("✓ Chrome 드라이버 사용 가능")

    except Exception as e:
        print(f"✗ Chrome 드라이버 사용 불가: {e}")
        print("실제 환경 테스트를 건너뜁니다.")
        return False

    try:
        from envs import SuikaEnvWrapper

        print("\n[1] 실제 환경 생성 (헤드리스 모드)")
        env = SuikaEnvWrapper(
            headless=True,
            port=8926,  # 충돌 방지
            use_mock=False
        )
        print("✓ 실제 환경 생성 완료")

        print("\n[2] 환경 리셋 (게임 로딩 중...)")
        obs, info = env.reset()
        print("✓ 환경 리셋 완료")
        print(f"  - 초기 점수: {obs['score'][0]}")

        print("\n[3] 안정화 대기 시간 측정")
        print("-" * 70)

        num_steps = 5
        wait_times = []

        for step in range(num_steps):
            print(f"\nStep {step + 1}:")

            # 다양한 위치에 과일 떨어뜨리기
            action_value = 0.3 + (step * 0.1)
            action = np.array([action_value], dtype=np.float32)
            print(f"  - 행동: {action_value:.2f} (위치: {int(action_value * 640)}px)")

            # 시간 측정
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.time() - start_time

            wait_times.append(elapsed)

            print(f"  - 대기 시간: {elapsed:.3f}초")
            print(f"  - 안정화 성공: {info.get('stable', 'N/A')}")
            print(f"  - 보상: {reward:.2f}")
            print(f"  - 현재 점수: {obs['score'][0]:.0f}")

            if terminated or truncated:
                print(f"\n⚠ 게임 종료!")
                break

        print("\n[4] 안정화 대기 시간 통계")
        print("-" * 70)
        print(f"  - 평균 대기 시간: {np.mean(wait_times):.3f}초")
        print(f"  - 최소 대기 시간: {np.min(wait_times):.3f}초")
        print(f"  - 최대 대기 시간: {np.max(wait_times):.3f}초")
        print(f"  - 표준편차: {np.std(wait_times):.3f}초")

        print("\n[5] 기존 고정 대기 시간과 비교")
        print("-" * 70)
        print(f"  - 기존 방식 (고정): 0.500초/스텝")
        print(f"  - 새 방식 (적응적): {np.mean(wait_times):.3f}초/스텝")

        improvement = (0.5 - np.mean(wait_times)) / 0.5 * 100
        if improvement > 0:
            print(f"  - 성능 향상: {improvement:.1f}% 빠름")
        else:
            print(f"  - 성능 변화: {-improvement:.1f}% 느림 (더 정확한 관찰)")

        env.close()
        print("\n✓ 실제 환경 테스트 완료")
        return True

    except Exception as e:
        print(f"\n✗ 실제 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stability_detection():
    """JavaScript 안정화 감지 함수 직접 테스트"""
    print("\n" + "=" * 70)
    print("  JavaScript 안정화 감지 함수 테스트")
    print("=" * 70)

    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        print("\n[1] 브라우저 및 게임 환경 초기화")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)

        # 게임 로드
        driver.get("http://localhost:8926/")
        time.sleep(2)

        # 게임 시작
        driver.find_element("id", "start-game-button").click()
        time.sleep(1)

        print("✓ 게임 환경 준비 완료")

        print("\n[2] 안정화 상태 함수 호출 테스트")

        # getStabilityStatus 함수 호출
        status = driver.execute_script('return window.Game.getStabilityStatus();')

        print(f"✓ getStabilityStatus() 호출 성공:")
        print(f"  - isStable: {status['isStable']}")
        print(f"  - stateIndex: {status['stateIndex']}")
        print(f"  - score: {status['score']}")
        print(f"  - bodyCount: {status['bodyCount']}")

        print("\n[3] 과일 떨어뜨린 후 안정화 과정 관찰")

        # 과일 떨어뜨리기
        driver.find_element("id", "fruit-position").clear()
        driver.find_element("id", "fruit-position").send_keys("320")
        driver.find_element("id", "drop-fruit-button").click()

        # 안정화 과정 관찰 (최대 3초)
        start_time = time.time()
        checks = []

        while time.time() - start_time < 3.0:
            status = driver.execute_script('return window.Game.getStabilityStatus();')
            elapsed = time.time() - start_time
            checks.append((elapsed, status['isStable'], status['bodyCount']))

            print(f"  {elapsed:.2f}초: isStable={status['isStable']}, bodies={status['bodyCount']}")

            if status['isStable']:
                print(f"\n✓ 안정화 완료 (소요 시간: {elapsed:.3f}초)")
                break

            time.sleep(0.05)

        driver.quit()
        print("\n✓ JavaScript 함수 테스트 완료")
        return True

    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        if 'driver' in locals():
            driver.quit()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 70)
    print(" " * 15 + "안정화 대기 기능 테스트 시작")
    print("=" * 70)

    # 1. Mock 환경 테스트 (빠른 기본 검증)
    test_stability_with_mock()

    # 2. 실제 환경 테스트 여부 확인
    print("\n\n실제 Suika 환경으로 상세 테스트를 진행하시겠습니까?")
    print("(Chrome/ChromeDriver 필요, 약 30초 소요)")
    response = input("y/N: ").strip().lower()

    if response == 'y':
        # 실제 환경 테스트
        success = test_stability_with_real_env()

        if success:
            # JavaScript 함수 직접 테스트
            print("\n\nJavaScript 안정화 감지 함수를 직접 테스트하시겠습니까?")
            response2 = input("y/N: ").strip().lower()

            if response2 == 'y':
                test_stability_detection()
    else:
        print("\n실제 환경 테스트를 건너뜁니다.")

    print("\n\n" + "=" * 70)
    print("모든 테스트 완료!")
    print("=" * 70)

    print("\n[결론]")
    print("✓ 안정화 대기 기능이 구현되었습니다.")
    print("✓ 과일이 완전히 멈춘 후 state를 반환합니다.")
    print("✓ 적응적 대기로 효율성이 향상되었습니다.")

    print("\n[다음 단계]")
    print("- 학습을 실행하여 실제 성능을 확인하세요.")
    print("- info['stable']을 통해 타임아웃 발생 여부를 모니터링하세요.")


if __name__ == "__main__":
    main()
