"""
후반부 느려짐 원인 조사

에피소드 진행에 따라 물리 객체 수, 시뮬레이션 시간 등을 추적합니다.
"""

import sys
import os
import time
import numpy as np

# suika_rl 패키지 경로 추가
suika_rl_path = os.path.join(os.path.dirname(__file__), 'suika_rl')
if os.path.exists(suika_rl_path) and suika_rl_path not in sys.path:
    sys.path.insert(0, suika_rl_path)

from suika_env.suika_browser_env import SuikaBrowserEnv


def profile_with_physics_stats(env, num_steps=60):
    """
    물리 통계와 함께 성능을 측정합니다.
    """
    print("\nReset...")
    obs, info = env.reset()

    step_data = []

    for step_idx in range(num_steps):
        action = env.action_space.sample()

        # Step 전 통계 가져오기
        pre_stats = env.driver.execute_script('''
            const bodies = Matter.Composite.allBodies(window.engine.world);
            const dynamicBodies = bodies.filter(b => !b.isStatic);
            return {
                totalBodies: bodies.length,
                dynamicBodies: dynamicBodies.length,
                staticBodies: bodies.filter(b => b.isStatic).length
            };
        ''')

        # Step 실행
        step_start = time.time()
        obs, reward, terminated, truncated, info = env.step(action)
        step_time = time.time() - step_start

        # Step 후 통계
        post_stats = env.driver.execute_script('''
            const bodies = Matter.Composite.allBodies(window.engine.world);
            const dynamicBodies = bodies.filter(b => !b.isStatic);
            return {
                totalBodies: bodies.length,
                dynamicBodies: dynamicBodies.length,
                staticBodies: bodies.filter(b => b.isStatic).length
            };
        ''')

        step_data.append({
            'step': step_idx + 1,
            'time': step_time,
            'pre_dynamic': pre_stats['dynamicBodies'],
            'post_dynamic': post_stats['dynamicBodies'],
            'pre_total': pre_stats['totalBodies'],
            'post_total': post_stats['totalBodies'],
            'reward': reward,
            'score': info.get('score', 0)
        })

        # 5 스텝마다 출력
        if (step_idx + 1) % 5 == 0:
            recent = step_data[-5:]
            avg_time = np.mean([d['time'] for d in recent])
            avg_bodies = np.mean([d['post_dynamic'] for d in recent])
            print(f"Steps {step_idx-3:2d}-{step_idx+1:2d}: "
                  f"{avg_time:.4f}s/step, "
                  f"{avg_bodies:.1f} dynamic bodies, "
                  f"score={info.get('score', 0)}")

        if terminated or truncated:
            print(f"\nEpisode terminated at step {step_idx+1}")
            break

    return step_data


def analyze_late_episode_slowdown(step_data):
    """
    후반부 느려짐을 분석합니다.
    """
    print("\n" + "="*70)
    print("LATE EPISODE SLOWDOWN ANALYSIS")
    print("="*70)

    # 초반 30%, 중반 40%, 후반 30%로 나눔
    n = len(step_data)
    cutoff1 = int(n * 0.3)
    cutoff2 = int(n * 0.7)

    early = step_data[:cutoff1]
    mid = step_data[cutoff1:cutoff2]
    late = step_data[cutoff2:]

    def stats(data, name):
        times = [d['time'] for d in data]
        bodies = [d['post_dynamic'] for d in data]
        scores = [d['score'] for d in data]

        print(f"\n{name}:")
        print(f"  Steps: {len(data)}")
        print(f"  Avg time: {np.mean(times):.4f}s")
        print(f"  Avg dynamic bodies: {np.mean(bodies):.1f}")
        print(f"  Avg score: {np.mean(scores):.1f}")
        print(f"  Max time: {np.max(times):.4f}s")
        print(f"  Max bodies: {int(np.max(bodies))}")

    stats(early, "Early Episode (first 30%)")
    stats(mid, "Mid Episode (30-70%)")
    stats(late, "Late Episode (last 30%)")

    # 상관관계 분석
    times = [d['time'] for d in step_data]
    bodies = [d['post_dynamic'] for d in step_data]
    steps = [d['step'] for d in step_data]

    # numpy 상관계수
    corr_time_bodies = np.corrcoef(times, bodies)[0, 1]
    corr_time_steps = np.corrcoef(times, steps)[0, 1]

    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)
    print(f"Time vs Bodies count: {corr_time_bodies:.3f}")
    print(f"Time vs Step number: {corr_time_steps:.3f}")

    if corr_time_bodies > 0.5:
        print("\nConclusion: Strong positive correlation between body count and step time.")
        print("More bodies = slower physics simulation.")
    elif corr_time_steps > 0.5:
        print("\nConclusion: Strong positive correlation between step number and step time.")
        print("This suggests accumulating overhead (memory leaks, garbage collection, etc.)")
    else:
        print("\nConclusion: No strong correlation found.")


def main():
    print("="*70)
    print("LATE EPISODE SLOWDOWN INVESTIGATION")
    print("="*70)

    env = SuikaBrowserEnv(
        headless=True,
        port=8925,
        fast_mode=True
    )

    print("\nRunning 60 steps with detailed physics tracking...")

    step_data = profile_with_physics_stats(env, num_steps=60)
    analyze_late_episode_slowdown(step_data)

    env.close()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
