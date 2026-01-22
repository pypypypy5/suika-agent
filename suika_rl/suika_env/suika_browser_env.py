from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import gymnasium
import ipdb
import io
import numpy as np    
from PIL import Image
import imageio
import subprocess
import socket
import os

class SuikaBrowserEnv(gymnasium.Env):
    def __init__(self, headless=True, port=8923, delay_before_img_capture=0.5, fast_mode=True) -> None:
        self.game_url = f"http://localhost:{port}/"
        # Check if port is already in use
        self.server = None
        if not self.is_port_in_use(port):
            # Get the absolute path of the current script
            script_dir = os.path.dirname(os.path.realpath(__file__))
            # Construct the absolute path of the suika-game directory
            suika_game_dir = os.path.join(script_dir, 'suika-game')
            self.server = subprocess.Popen(["python", "-m", "http.server", str(port)], cwd=suika_game_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        opts = webdriver.ChromeOptions()
        opts.add_argument("--width=1024")
        opts.add_argument("--height=768")
        self.headless = headless
        if headless:
            opts.add_argument("--headless=new")
        self.delay_before_img_capture = delay_before_img_capture
        self.fast_mode = fast_mode
        self.img_width = 128
        self.img_height = 128
        self.driver = webdriver.Chrome(options=opts)
        # NOTE: Image shape is (128, 128, 3) RGB, not RGBA
        # PIL may convert RGBA to RGB during processing
        _obs_dict = {
            'image': gymnasium.spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3),  dtype="uint8"),
            'score': gymnasium.spaces.Box(low=0, high=1000000, shape=(1,), dtype="float32"),
        }
        self.observation_space = gymnasium.spaces.Dict(_obs_dict)
        self.action_space = gymnasium.spaces.Box(low=0, high=1, shape=(1,))

    def reset(self,seed=None, options=None):
        self._reload()

        # Enable fast mode if configured
        if self.fast_mode:
            self.driver.execute_script('window.Game.setFastMode(true);')
            # Render once for initial screenshot
            self.driver.execute_script('window.Render.world(window.render);')

        info = {}
        self.score = 0
        obs, status = self._get_obs_and_status()
        return obs, info

    def _reload(self):
        # open the game.
        self.driver.get(self.game_url)
        # click start game button with id "start-game-button"
        self.driver.find_element(By.ID, 'start-game-button').click()
        time.sleep(1)
    
    def _get_obs_and_status(self):
        img = self._capture_canvas()
        status, score = self.driver.execute_script('return [window.Game.stateIndex, window.Game.score];')
        score = np.array([score], dtype=np.float32)
        return dict(image=img, score=score), status
    
    def _capture_canvas(self):
        # OPTIMIZATION: Screenshots are a major bottleneck. We optimize by:
        # 1. Getting the canvas element once (caching would be even better but element can become stale)
        # 2. Using BILINEAR instead of LANCZOS for faster (though slightly lower quality) resizing
        # 3. Converting directly to array without intermediate steps where possible
        canvas = self.driver.find_element(By.ID, 'game-canvas')
        image_string = canvas.screenshot_as_png
        img = Image.open(io.BytesIO(image_string))
        # first crop out right hand side and lower bar.
        img = img.crop((0, 0, 520, img.height))

        # Use faster BILINEAR filter instead of LANCZOS (LANCZOS is higher quality but slower)
        imgResized = img.resize((self.img_width, self.img_height), Image.Resampling.BILINEAR)
        arr = np.asarray(imgResized)
        return arr

    def _wait_until_stable_fast(self, max_steps=300, delta_ms=16.67, threshold=0.01):
        """
        Wait until stable using fast-forward physics simulation (fast mode only).
        No real-time waiting - physics steps are executed as fast as possible.

        Args:
            max_steps: Maximum physics steps to simulate (default: 300 = ~5 seconds simulated time)
            delta_ms: Time delta per physics step in ms (default: 16.67 = 60 FPS)
            threshold: Velocity threshold for stability (default: 0.01)

        Returns:
            dict: Result with 'success', 'steps', and 'time' keys
        """
        try:
            result = self.driver.execute_script(f'''
                return window.Game.fastForwardUntilStable({max_steps}, {delta_ms}, {threshold});
            ''')
            return result
        except Exception as e:
            print(f"Warning: Fast mode failed: {e}")
            return {'success': False, 'steps': 0, 'time': 0}

    def _wait_until_stable(self, max_wait_time=5.0, check_interval=0.05, stable_duration=0.2):
        """
        Wait until all fruits have stopped moving (velocities near zero).

        This ensures that the observation returned to the agent represents a stable state
        where all physics interactions (collisions, merges) have completed.

        Automatically uses fast mode if enabled, otherwise uses real-time mode.

        Args:
            max_wait_time: Maximum time to wait in seconds (default: 5.0)
            check_interval: How often to check stability in seconds (default: 0.05)
            stable_duration: How long the state must remain stable in seconds (default: 0.2)

        Returns:
            bool: True if stable, False if timed out
        """
        # Fast mode: no real-time waiting
        if self.fast_mode:
            max_steps = int(max_wait_time * 1000 / 16.67)  # Convert seconds to physics steps
            result = self._wait_until_stable_fast(max_steps=max_steps, delta_ms=16.67, threshold=0.01)
            return result['success']

        # Real-time mode: poll and wait
        start_time = time.time()
        stable_start = None

        while time.time() - start_time < max_wait_time:
            # Check stability status from JavaScript
            try:
                status = self.driver.execute_script('return window.Game.getStabilityStatus();')

                if status['isStable']:
                    if stable_start is None:
                        stable_start = time.time()
                    elif time.time() - stable_start >= stable_duration:
                        # Stable state maintained for required duration
                        return True
                else:
                    # Reset if became unstable again
                    stable_start = None

                time.sleep(check_interval)

            except Exception as e:
                # If JavaScript function not available, fall back to fixed delay
                print(f"Warning: Could not check stability status: {e}")
                time.sleep(self.delay_before_img_capture)
                return False

        # Timeout reached
        print(f"Warning: Stability timeout after {max_wait_time}s")
        return False

    
    def step(self, action):
        driver = self.driver
        action = action[0]
        info = {}
        # action is a float from 0 to 1. need to convert to int from 0 to 640.
        action_x = int(action * 640)

        # OPTIMIZATION: Use direct JavaScript execution instead of Selenium DOM manipulation
        # This is ~3x faster than finding elements and clicking them
        driver.execute_script(f'window.Game.addFruit({action_x});')

        # Wait until all fruits have stopped moving
        stable = self._wait_until_stable(
            max_wait_time=5.0,
            check_interval=0.05,
            stable_duration=0.2
        )
        info['stable'] = stable

        obs, status = self._get_obs_and_status()
        reward = 0
        # check if game is over.
        terminal = status == 3
        truncated = False
        score = obs['score'].item()
        info['score'] = score
        reward += score - self.score
        self.score = score

        return obs, reward, terminal, truncated, info


    def is_port_in_use(self, port):
        """Check if a given port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def close(self):
        super().close()
        if self.driver is not None:
            self.driver.quit()
        # Stop the server
        if self.server is not None:
            self.server.terminate()

if __name__ == "__main__":
    env = SuikaBrowserEnv(headless=False, delay_before_img_capture=0.5)
    try:
        video = []
        obs, info = env.reset()
        # video.append(obs['image'])
        # import imageio
        terminated = False
        while not terminated:
            action = [0]
            obs, rew, terminated, truncated, info = env.step(action)
            # video.append(obs['image'])
            if terminated:
                break
    finally:
        env.close()