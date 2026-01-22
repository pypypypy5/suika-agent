"""
환경 성능 프로파일링 스크립트

게임 환경의 병목 지점을 찾기 위한 상세 프로파일링을 수행합니다.
"""

import sys
import os
import time
import numpy as np
from collections import defaultdict

# suika_rl 패키지 경로 추가
suika_rl_path = os.path.join(os.path.dirname(__file__), 'suika_rl')
if os.path.exists(suika_rl_path) and suika_rl_path not in sys.path:
    sys.path.insert(0, suika_rl_path)

from envs.suika_wrapper import SuikaEnvWrapper


def profile_episode(env, num_steps=50, episode_num=1):
    """
    한 에피소드 동안의 성능을 프로파일링합니다.

    Args:
        env: 환경 인스턴스
        num_steps: 프로파일링할 스텝 수
        episode_num: 에피소드 번호 (로깅용)

    Returns:
        Dict containing profiling results
    """
    timing_data = {
        'reset': [],
        'step': [],
        'step_breakdown': defaultdict(list),
        'late_episode_steps': []  # 후반부 스텝들의 시간
    }

    # Reset 시간 측정
    start = time.time()
    obs, info = env.reset()
    reset_time = time.time() - start
    timing_data['reset'].append(reset_time)

    print(f"\nEpisode {episode_num}:")
    print(f"  Reset time: {reset_time:.4f}s")

    # Step 시간 측정
    step_times = []
    for step_idx in range(num_steps):
        # 랜덤 액션
        action = env.action_space.sample()

        # 전체 step 시간
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start

        step_times.append(step_time)
        timing_data['step'].append(step_time)

        # 후반부 스텝 (전체의 70% 이후) 따로 기록
        if step_idx >= num_steps * 0.7:
            timing_data['late_episode_steps'].append(step_time)

        # 매 10 스텝마다 출력
        if (step_idx + 1) % 10 == 0:
            recent_avg = np.mean(step_times[-10:])
            print(f"  Steps {step_idx-8}-{step_idx+1}: avg {recent_avg:.4f}s/step")

        if terminated or truncated:
            print(f"  Episode terminated at step {step_idx+1}")
            break

    return timing_data


def analyze_timing_data(all_timing_data):
    """
    수집된 타이밍 데이터를 분석합니다.
    """
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)

    # Reset 시간
    reset_times = []
    for data in all_timing_data:
        reset_times.extend(data['reset'])

    print(f"\nReset Performance:")
    print(f"  Average: {np.mean(reset_times):.4f}s")
    print(f"  Std Dev: {np.std(reset_times):.4f}s")
    print(f"  Min: {np.min(reset_times):.4f}s")
    print(f"  Max: {np.max(reset_times):.4f}s")

    # Step 시간 - 전체
    all_step_times = []
    for data in all_timing_data:
        all_step_times.extend(data['step'])

    print(f"\nStep Performance (Overall):")
    print(f"  Average: {np.mean(all_step_times):.4f}s")
    print(f"  Std Dev: {np.std(all_step_times):.4f}s")
    print(f"  Min: {np.min(all_step_times):.4f}s")
    print(f"  Max: {np.max(all_step_times):.4f}s")
    print(f"  Median: {np.median(all_step_times):.4f}s")

    # 후반부 스텝 분석
    all_late_steps = []
    for data in all_timing_data:
        all_late_steps.extend(data['late_episode_steps'])

    if all_late_steps:
        print(f"\nLate Episode Steps (last 30%):")
        print(f"  Average: {np.mean(all_late_steps):.4f}s")
        print(f"  Std Dev: {np.std(all_late_steps):.4f}s")
        print(f"  Max: {np.max(all_late_steps):.4f}s")

        # 초반 vs 후반 비교
        early_steps = []
        for data in all_timing_data:
            steps = data['step']
            cutoff = int(len(steps) * 0.3)
            if cutoff > 0:
                early_steps.extend(steps[:cutoff])

        if early_steps:
            early_avg = np.mean(early_steps)
            late_avg = np.mean(all_late_steps)
            slowdown = (late_avg - early_avg) / early_avg * 100

            print(f"\n  Early episode avg: {early_avg:.4f}s")
            print(f"  Late episode avg: {late_avg:.4f}s")
            print(f"  Slowdown: {slowdown:.1f}%")

            if slowdown > 20:
                print(f"  ⚠️  WARNING: Significant late-episode slowdown detected!")

    # 통계 요약
    total_steps = len(all_step_times)
    total_time = sum(all_step_times)

    print(f"\nOverall Statistics:")
    print(f"  Total steps: {total_steps}")
    print(f"  Total step time: {total_time:.2f}s")
    print(f"  Steps per second: {total_steps / total_time:.2f}")

    return {
        'reset_avg': np.mean(reset_times),
        'step_avg': np.mean(all_step_times),
        'step_median': np.median(all_step_times),
        'late_step_avg': np.mean(all_late_steps) if all_late_steps else None,
        'total_steps': total_steps,
        'steps_per_second': total_steps / total_time
    }


def main():
    """메인 프로파일링 루틴"""
    print("="*60)
    print("ENVIRONMENT PERFORMANCE PROFILING")
    print("="*60)

    # 환경 설정
    print("\nInitializing environment...")
    env = SuikaEnvWrapper(
        headless=True,
        port=8923,
        fast_mode=True,  # fast_mode 사용
        use_mock=False
    )

    print("Environment initialized successfully!")

    # 프로파일링 설정
    num_episodes = 3
    steps_per_episode = 50

    print(f"\nProfiling Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Fast mode: True")

    # 각 에피소드 프로파일링
    all_timing_data = []

    for ep in range(num_episodes):
        timing_data = profile_episode(env, steps_per_episode, ep + 1)
        all_timing_data.append(timing_data)

        # 에피소드 간 짧은 대기
        time.sleep(0.5)

    # 결과 분석
    results = analyze_timing_data(all_timing_data)

    # 환경 종료
    env.close()

    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)

    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user.")
    except Exception as e:
        print(f"\n\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
