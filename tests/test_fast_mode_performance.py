"""
고속 모드 성능 테스트

실시간 모드와 고속 모드의 성능을 비교합니다.
"""

import sys
import os
import time
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_performance_comparison():
    """실시간 모드 vs 고속 모드 성능 비교"""

    # Chrome 드라이버 확인
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        print("=" * 70)
        print("  고속 모드 성능 테스트")
        print("=" * 70)

        print("\n[준비] Chrome 드라이버 확인")
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=options)
        driver.quit()
        print("✓ Chrome 드라이버 사용 가능\n")

    except Exception as e:
        print(f"✗ Chrome 드라이버 사용 불가: {e}")
        print("이 테스트를 실행하려면 Chrome과 ChromeDriver가 필요합니다.")
        return

    from envs import SuikaEnvWrapper

    num_steps = 20
    results = {}

    # ===== 실시간 모드 테스트 =====
    print("=" * 70)
    print("  [1] 실시간 모드 테스트 (fast_mode=False)")
    print("=" * 70)

    try:
        env = SuikaEnvWrapper(
            headless=True,
            port=8927,
            fast_mode=False,  # 실시간 모드
            use_mock=False
        )

        print("\n환경 리셋 중...")
        obs, info = env.reset()
        print("✓ 리셋 완료\n")

        step_times = []
        total_start = time.time()

        for step in range(num_steps):
            action = np.array([0.4 + (step % 5) * 0.1], dtype=np.float32)

            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start

            step_times.append(step_time)

            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1:2d}: {step_time:.3f}초")

            if terminated or truncated:
                print(f"\n게임 종료 (step {step + 1})")
                break

        total_time = time.time() - total_start

        results['realtime'] = {
            'step_times': step_times,
            'total_time': total_time,
            'steps': len(step_times),
            'avg_step_time': np.mean(step_times),
            'min_step_time': np.min(step_times),
            'max_step_time': np.max(step_times),
            'std_step_time': np.std(step_times)
        }

        print(f"\n[실시간 모드 결과]")
        print(f"  총 시간: {total_time:.2f}초")
        print(f"  완료 step: {len(step_times)}")
        print(f"  평균 step 시간: {results['realtime']['avg_step_time']:.3f}초")
        print(f"  최소/최대: {results['realtime']['min_step_time']:.3f}초 / {results['realtime']['max_step_time']:.3f}초")

        env.close()
        print("\n✓ 실시간 모드 테스트 완료")

    except Exception as e:
        print(f"\n✗ 실시간 모드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results['realtime'] = None

    # 환경 전환을 위해 잠시 대기
    time.sleep(2)

    # ===== 고속 모드 테스트 =====
    print("\n\n" + "=" * 70)
    print("  [2] 고속 모드 테스트 (fast_mode=True)")
    print("=" * 70)

    try:
        env = SuikaEnvWrapper(
            headless=True,
            port=8928,
            fast_mode=True,  # 고속 모드
            use_mock=False
        )

        print("\n환경 리셋 중...")
        obs, info = env.reset()
        print("✓ 리셋 완료\n")

        step_times = []
        total_start = time.time()

        for step in range(num_steps):
            action = np.array([0.4 + (step % 5) * 0.1], dtype=np.float32)

            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start

            step_times.append(step_time)

            if (step + 1) % 5 == 0:
                print(f"  Step {step + 1:2d}: {step_time:.3f}초")

            if terminated or truncated:
                print(f"\n게임 종료 (step {step + 1})")
                break

        total_time = time.time() - total_start

        results['fast'] = {
            'step_times': step_times,
            'total_time': total_time,
            'steps': len(step_times),
            'avg_step_time': np.mean(step_times),
            'min_step_time': np.min(step_times),
            'max_step_time': np.max(step_times),
            'std_step_time': np.std(step_times)
        }

        print(f"\n[고속 모드 결과]")
        print(f"  총 시간: {total_time:.2f}초")
        print(f"  완료 step: {len(step_times)}")
        print(f"  평균 step 시간: {results['fast']['avg_step_time']:.3f}초")
        print(f"  최소/최대: {results['fast']['min_step_time']:.3f}초 / {results['fast']['max_step_time']:.3f}초")

        env.close()
        print("\n✓ 고속 모드 테스트 완료")

    except Exception as e:
        print(f"\n✗ 고속 모드 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results['fast'] = None

    # ===== 결과 비교 =====
    print("\n\n" + "=" * 70)
    print("  [3] 성능 비교 분석")
    print("=" * 70)

    if results['realtime'] and results['fast']:
        rt = results['realtime']
        ft = results['fast']

        speedup = rt['avg_step_time'] / ft['avg_step_time']
        total_speedup = rt['total_time'] / ft['total_time']

        print(f"\n{'항목':<20s} | {'실시간 모드':<15s} | {'고속 모드':<15s} | 개선")
        print("-" * 70)
        print(f"{'평균 step 시간':<20s} | {rt['avg_step_time']:>12.3f}초 | {ft['avg_step_time']:>12.3f}초 | {speedup:>5.1f}x")
        print(f"{'총 실행 시간':<20s} | {rt['total_time']:>12.2f}초 | {ft['total_time']:>12.2f}초 | {total_speedup:>5.1f}x")
        print(f"{'시간당 step 수':<20s} | {3600/rt['avg_step_time']:>12.0f} | {3600/ft['avg_step_time']:>12.0f} | {speedup:>5.1f}x")

        print(f"\n[학습 시간 예측]")
        steps_to_train = [10000, 50000, 100000]
        print(f"\n{'학습 step 수':<15s} | {'실시간 모드':<20s} | {'고속 모드':<20s} | 시간 절약")
        print("-" * 70)
        for steps in steps_to_train:
            rt_hours = (steps * rt['avg_step_time']) / 3600
            ft_hours = (steps * ft['avg_step_time']) / 3600
            saved_hours = rt_hours - ft_hours

            rt_str = f"{rt_hours:.1f}시간" if rt_hours < 24 else f"{rt_hours/24:.1f}일"
            ft_str = f"{ft_hours:.1f}시간" if ft_hours < 24 else f"{ft_hours/24:.1f}일"
            saved_str = f"{saved_hours:.1f}시간" if saved_hours < 24 else f"{saved_hours/24:.1f}일"

            print(f"{steps:>15,} | {rt_str:>20s} | {ft_str:>20s} | {saved_str}")

        print(f"\n[결론]")
        print(f"✓ 고속 모드가 평균 {speedup:.1f}배 빠릅니다")
        print(f"✓ 렌더링 비활성화로 추가 성능 향상")
        print(f"✓ 학습 시 fast_mode=True 사용 권장")

        # 성능 상세 분석
        print(f"\n[성능 상세 분석]")
        print(f"\n실시간 모드:")
        print(f"  - 표준편차: {rt['std_step_time']:.3f}초")
        print(f"  - 변동 계수: {rt['std_step_time']/rt['avg_step_time']*100:.1f}%")

        print(f"\n고속 모드:")
        print(f"  - 표준편차: {ft['std_step_time']:.3f}초")
        print(f"  - 변동 계수: {ft['std_step_time']/ft['avg_step_time']*100:.1f}%")

    else:
        print("\n일부 테스트가 실패하여 비교를 수행할 수 없습니다.")

    print("\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)


def main():
    """메인 테스트 실행"""
    print("\n고속 모드 성능 테스트를 시작합니다.")
    print("주의: 이 테스트는 약 1~2분 소요됩니다.\n")

    response = input("계속하시겠습니까? (y/N): ").strip().lower()

    if response == 'y':
        test_performance_comparison()
    else:
        print("\n테스트를 취소했습니다.")


if __name__ == "__main__":
    main()
