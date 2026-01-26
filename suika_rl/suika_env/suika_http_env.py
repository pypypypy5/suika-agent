"""
Suika Game HTTP Environment
Connects to Node.js game server via HTTP instead of Selenium
"""

import gymnasium
import numpy as np
import requests
import time
from typing import Optional, Tuple, Dict, Any


class SuikaBrowserEnv(gymnasium.Env):
    """
    HTTP-based Suika Game environment.

    This is a drop-in replacement for the Selenium-based SuikaBrowserEnv.
    It maintains the same interface but communicates with a Node.js server
    instead of controlling a Chrome browser.

    Note: The class name is kept as 'SuikaBrowserEnv' for backward compatibility.
    """

    def __init__(
        self,
        headless: bool = True,
        port: int = 8924,
        delay_before_img_capture: float = 0.5,
        fast_mode: bool = True,
        server_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ) -> None:
        """
        Initialize the HTTP-based Suika Game environment.

        Args:
            headless: Ignored (kept for compatibility)
            port: Server port (default: 8924)
            delay_before_img_capture: Ignored (kept for compatibility)
            fast_mode: Ignored (server always uses fast mode)
            server_url: Full server URL (overrides port if provided)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        # Server configuration
        if server_url:
            self.server_url = server_url.rstrip('/')
        else:
            self.server_url = f"http://localhost:{port}"

        self.api_base = f"{self.server_url}/api"
        self.timeout = timeout
        self.max_retries = max_retries

        # Generate unique game ID for this environment instance
        self.game_id = f"game_{id(self)}"

        # Image dimensions (260x384 maintains 2:3 aspect ratio)
        self.img_width = 260
        self.img_height = 384

        # Score tracking
        self.score = 0

        # Define observation and action spaces
        self.observation_space = gymnasium.spaces.Dict({
            'image': gymnasium.spaces.Box(
                low=0, high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            ),
            'score': gymnasium.spaces.Box(
                low=0, high=1000000,
                shape=(1,),
                dtype=np.float32
            ),
        })

        self.action_space = gymnasium.spaces.Box(
            low=0, high=1,
            shape=(1,),
            dtype=np.float32
        )

        # Check server health
        self._check_server_health()

    def _check_server_health(self) -> None:
        """Check if the game server is running."""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5.0
            )
            response.raise_for_status()
            print(f"Connected to Suika Game Server at {self.server_url}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Cannot connect to game server at {self.server_url}")
            print(f"Error: {e}")
            print("Make sure the server is running: cd suika_rl/server && npm start")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint (e.g., '/reset')
            data: Request data (for POST)
            retry_count: Current retry attempt

        Returns:
            Response JSON data

        Raises:
            RuntimeError: If request fails after max retries
        """
        url = f"{self.api_base}{endpoint}"

        try:
            if method == 'GET':
                response = requests.get(url, timeout=self.timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                print(f"Request failed (attempt {retry_count + 1}/{self.max_retries}): {e}")
                time.sleep(0.5 * (retry_count + 1))  # Exponential backoff
                return self._make_request(method, endpoint, data, retry_count + 1)
            else:
                raise RuntimeError(f"Request failed after {self.max_retries} retries: {e}")

    def _parse_observation(self, obs_data: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Parse observation data from server response.

        Args:
            obs_data: Observation dict from server

        Returns:
            Tuple of (observation dict, done flag)
        """
        # Parse image (RGBA array from server -> RGB for compatibility)
        raw_image = np.array(obs_data['image'])
        # Some servers may send packed 32-bit RGBA; detect and unpack
        if raw_image.dtype != np.uint8 or (raw_image.size > 0 and raw_image.max() > 255):
            if raw_image.size == self.img_height * self.img_width:
                packed = raw_image.astype(np.uint32)
                rgba = np.empty((packed.size, 4), dtype=np.uint8)
                rgba[:, 0] = packed & 0xFF
                rgba[:, 1] = (packed >> 8) & 0xFF
                rgba[:, 2] = (packed >> 16) & 0xFF
                rgba[:, 3] = (packed >> 24) & 0xFF
                image_rgba = rgba.reshape(self.img_height, self.img_width, 4)
            else:
                image_array = np.clip(raw_image, 0, 255).astype(np.uint8)
                image_rgba = image_array.reshape(self.img_height, self.img_width, 4)
        else:
            image_array = raw_image.astype(np.uint8)
            # Reshape from flat array to (height, width, 4) RGBA
            image_rgba = image_array.reshape(self.img_height, self.img_width, 4)

        # Convert RGBA to RGB (drop alpha channel)
        image_rgb = image_rgba[:, :, :3]

        # Parse score
        score = np.array([obs_data['score']], dtype=np.float32)

        # Parse done flag
        done = obs_data['done']

        observation = {
            'image': image_rgb,
            'score': score
        }

        return observation, done

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        # Call reset API
        data = {
            'game_id': self.game_id,
            'seed': seed
        }

        response = self._make_request('POST', '/reset', data)

        if not response.get('success'):
            raise RuntimeError(f"Reset failed: {response.get('error', 'Unknown error')}")

        # Parse observation
        obs_data = response['observation']
        observation, _ = self._parse_observation(obs_data)

        # Reset score tracking
        self.score = obs_data['score']

        info = {}
        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (float in [0, 1])

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action from [0, 1] to [0, 640]
        # Handle both 1D and 2D arrays
        if hasattr(action, 'shape') and len(action.shape) > 1:
            action_val = float(action[0][0])
        else:
            action_val = float(action[0])
        action_x = int(action_val * 640)

        # Call step API
        data = {
            'game_id': self.game_id,
            'action': action_x,
            'auto_reset': False  # Don't auto-reset on done
        }

        response = self._make_request('POST', '/step', data)

        if not response.get('success'):
            raise RuntimeError(f"Step failed: {response.get('error', 'Unknown error')}")

        # Parse observation
        obs_data = response['observation']
        observation, terminated = self._parse_observation(obs_data)

        # Calculate reward (score delta)
        current_score = obs_data['score']
        reward = current_score - self.score
        self.score = current_score

        # Info
        info = {
            'score': current_score,
            'stable': True  # HTTP version always waits for stability
        }

        # Truncated flag (not used in this version)
        truncated = False

        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Clean up environment.

        Optionally delete the game instance on the server to free memory.
        """
        try:
            # Delete game instance
            url = f"{self.server_url}/api/game/{self.game_id}"
            requests.delete(url, timeout=5.0)
        except requests.exceptions.RequestException:
            pass  # Ignore errors during cleanup

    def render(self, mode: str = 'rgb_array') -> Optional[np.ndarray]:
        """
        Render the environment (not implemented for HTTP version).

        Args:
            mode: Render mode

        Returns:
            None or RGB array
        """
        if mode == 'rgb_array':
            # Return last observation image
            # Note: This requires storing the last observation
            # For now, return None
            return None
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported")

    def is_port_in_use(self, port: int) -> bool:
        """
        Check if a port is in use (kept for compatibility).

        Args:
            port: Port number

        Returns:
            True if in use
        """
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
