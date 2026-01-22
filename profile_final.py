"""
최종 성능 측정 및 보고서 생성
"""

import sys
import os
import time
import numpy as np

# suika_rl 패키지 경로 추가
suika_rl_path = os.path.join(os.path.dirname(__file__), 'suika_rl')
if os.path.exists(suika_rl_path) and suika_rl_path not in sys.path:
    sys.path.insert(0, suika_rl_path)

from envs.suika_wrapper import SuikaEnvWrapper


def comprehensive_profile(env, num_episodes=5, steps_per_episode=50):
    """포괄적인 성능 프로파일링"""

    all_step_times = []
    all_reset_times = []
    episode_stats = []

    for ep in range(num_episodes):
        # Reset
        reset_start = time.time()
        obs, info = env.reset()
        reset_time = time.time() - reset_start
        all_reset_times.append(reset_time)

        step_times = []
        for step_idx in range(steps_per_episode):
            action = env.action_space.sample()

            step_start = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - step_start

            step_times.append(step_time)
            all_step_times.append(step_time)

            if terminated or truncated:
                break

        episode_stats.append({
            'episode': ep + 1,
            'steps': len(step_times),
            'avg_step_time': np.mean(step_times),
            'total_time': sum(step_times),
            'reset_time': reset_time
        })

        print(f"Episode {ep+1}: {len(step_times)} steps, "
              f"avg {np.mean(step_times):.4f}s/step, "
              f"reset {reset_time:.2f}s")

    return {
        'all_step_times': all_step_times,
        'all_reset_times': all_reset_times,
        'episode_stats': episode_stats
    }


def print_report(results, baseline=None):
    """최종 보고서 출력"""

    step_times = results['all_step_times']
    reset_times = results['all_reset_times']

    print("\n" + "="*70)
    print("FINAL PERFORMANCE REPORT")
    print("="*70)

    print("\nStep Performance:")
    print(f"  Average: {np.mean(step_times):.4f}s")
    print(f"  Median: {np.median(step_times):.4f}s")
    print(f"  Std Dev: {np.std(step_times):.4f}s")
    print(f"  Min: {np.min(step_times):.4f}s")
    print(f"  Max: {np.max(step_times):.4f}s")
    print(f"  Steps/second: {1.0/np.mean(step_times):.2f}")

    print("\nReset Performance:")
    print(f"  Average: {np.mean(reset_times):.4f}s")
    print(f"  Std Dev: {np.std(reset_times):.4f}s")

    print("\nThroughput:")
    total_steps = len(step_times)
    total_time = sum(step_times)
    print(f"  Total steps: {total_steps}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Overall throughput: {total_steps / total_time:.2f} steps/s")

    if baseline:
        print("\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)

        baseline_avg = baseline['step_avg']
        current_avg = np.mean(step_times)
        speedup = baseline_avg / current_avg
        improvement = (baseline_avg - current_avg) / baseline_avg * 100

        baseline_throughput = baseline['steps_per_second']
        current_throughput = total_steps / total_time
        throughput_gain = (current_throughput - baseline_throughput) / baseline_throughput * 100

        print(f"\nBaseline step time: {baseline_avg:.4f}s")
        print(f"Current step time: {current_avg:.4f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Improvement: {improvement:.1f}% faster")

        print(f"\nBaseline throughput: {baseline_throughput:.2f} steps/s")
        print(f"Current throughput: {current_throughput:.2f} steps/s")
        print(f"Throughput gain: {throughput_gain:.1f}%")

    # 후반부 분석
    n = len(step_times)
    early_cutoff = int(n * 0.3)
    late_cutoff = int(n * 0.7)

    early = step_times[:early_cutoff]
    late = step_times[late_cutoff:]

    if early and late:
        early_avg = np.mean(early)
        late_avg = np.mean(late)
        slowdown = (late_avg - early_avg) / early_avg * 100

        print("\n" + "="*70)
        print("LATE EPISODE ANALYSIS")
        print("="*70)
        print(f"Early episodes avg: {early_avg:.4f}s")
        print(f"Late episodes avg: {late_avg:.4f}s")
        print(f"Slowdown: {slowdown:.1f}%")


def main():
    print("="*70)
    print("FINAL PERFORMANCE MEASUREMENT")
    print("="*70)

    # Baseline 데이터 (최초 측정 결과)
    baseline = {
        'step_avg': 0.3611,
        'step_median': 0.3800,
        'steps_per_second': 2.77
    }

    print("\nBaseline (before optimization):")
    print(f"  Average step time: {baseline['step_avg']:.4f}s")
    print(f"  Throughput: {baseline['steps_per_second']:.2f} steps/s")

    print("\nInitializing optimized environment...")
    env = SuikaEnvWrapper(
        headless=True,
        port=8926,
        fast_mode=True,
        use_mock=False
    )

    print("\nRunning comprehensive profiling...")
    print("(5 episodes x 50 steps = 250 steps total)\n")

    results = comprehensive_profile(env, num_episodes=5, steps_per_episode=50)

    env.close()

    print_report(results, baseline=baseline)

    print("\n" + "="*70)
    print("PERFORMANCE MEASUREMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
