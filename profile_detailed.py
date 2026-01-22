"""
상세 프로파일링: step 내부의 각 부분별 시간 측정

Selenium 작업과 대기 시간을 분리해서 측정합니다.
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
from selenium.webdriver.common.by import By


class InstrumentedSuikaBrowserEnv(SuikaBrowserEnv):
    """타이밍 측정 기능을 추가한 환경"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timings = {
            'action_send': [],
            'wait_stable': [],
            'capture_canvas': [],
            'get_game_state': [],
            'total_step': []
        }

    def step(self, action):
        step_start = time.time()

        driver = self.driver
        action = action[0]
        info = {}

        # 1. 액션 전송
        t1 = time.time()
        action = str(int(action * 640))
        driver.find_element(By.ID, 'fruit-position').clear()
        driver.find_element(By.ID, 'fruit-position').send_keys(action)
        driver.find_element(By.ID, 'drop-fruit-button').click()
        t2 = time.time()
        self.timings['action_send'].append(t2 - t1)

        # 2. 안정화 대기
        t1 = time.time()
        stable = self._wait_until_stable(
            max_wait_time=5.0,
            check_interval=0.05,
            stable_duration=0.2
        )
        t2 = time.time()
        self.timings['wait_stable'].append(t2 - t1)
        info['stable'] = stable

        # 3. 화면 캡처
        t1 = time.time()
        img = self._capture_canvas()
        t2 = time.time()
        self.timings['capture_canvas'].append(t2 - t1)

        # 4. 게임 상태 가져오기
        t1 = time.time()
        status, score = self.driver.execute_script('return [window.Game.stateIndex, window.Game.score];')
        t2 = time.time()
        self.timings['get_game_state'].append(t2 - t1)

        score = np.array([score], dtype=np.float32)
        obs = dict(image=img, score=score)

        reward = 0
        terminal = status == 3
        truncated = False
        score_val = obs['score'].item()
        info['score'] = score_val
        reward += score_val - self.score
        self.score = score_val

        step_time = time.time() - step_start
        self.timings['total_step'].append(step_time)

        return obs, reward, terminal, truncated, info

    def print_timing_report(self):
        """타이밍 통계 출력"""
        print("\n" + "="*60)
        print("DETAILED TIMING BREAKDOWN")
        print("="*60)

        for key in ['action_send', 'wait_stable', 'capture_canvas', 'get_game_state', 'total_step']:
            if self.timings[key]:
                times = self.timings[key]
                avg = np.mean(times)
                std = np.std(times)
                min_t = np.min(times)
                max_t = np.max(times)

                print(f"\n{key}:")
                print(f"  Average: {avg:.4f}s ({avg/np.mean(self.timings['total_step'])*100:.1f}% of step)")
                print(f"  Std Dev: {std:.4f}s")
                print(f"  Min: {min_t:.4f}s")
                print(f"  Max: {max_t:.4f}s")


def main():
    print("="*60)
    print("DETAILED PERFORMANCE PROFILING")
    print("="*60)

    env = InstrumentedSuikaBrowserEnv(
        headless=True,
        port=8924,
        fast_mode=True
    )

    print("\nRunning 30 steps for detailed analysis...")

    obs, info = env.reset()

    for i in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    env.print_timing_report()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
