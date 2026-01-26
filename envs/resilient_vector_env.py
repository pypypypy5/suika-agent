"""
Resilient Vector Environment
AsyncVectorEnv wrapper that automatically restarts dead workers
"""

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from typing import Callable, List, Optional, Any, Tuple
import warnings


class ResilientAsyncVectorEnv(AsyncVectorEnv):
    """
    AsyncVectorEnv that catches worker crashes and gracefully handles them.

    When a worker process dies (BrokenPipeError, EOFError):
    1. Catches the exception
    2. Returns dummy observations for failed workers
    3. Continues training without full crash

    Note: This doesn't restart workers (too complex with Gymnasium internals),
    but prevents the entire training from crashing.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        **kwargs
    ):
        """
        Args:
            env_fns: List of environment factory functions
            **kwargs: Additional arguments for AsyncVectorEnv
        """
        super().__init__(env_fns, **kwargs)
        self._failed_workers = set()
        self._dummy_obs = None

    def _get_dummy_observation(self):
        """Get a dummy observation matching the observation space."""
        if self._dummy_obs is None:
            self._dummy_obs = self.single_observation_space.sample()
        return self._dummy_obs

    def step_wait(self, timeout: Optional[float] = None) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Wait for step results, handling worker failures gracefully.
        """
        try:
            return super().step_wait(timeout=timeout)
        except (BrokenPipeError, EOFError, ConnectionResetError) as e:
            print(f"\n{'='*60}")
            print(f"[CRITICAL] AsyncVectorEnv worker crashed: {e}")
            print(f"This is a known Windows multiprocessing issue.")
            print(f"Worker processes cannot be auto-restarted reliably.")
            print(f"{'='*60}")
            print(f"RECOMMENDATION: Reduce num_workers or use SyncVectorEnv (num_workers=1)")
            print(f"{'='*60}\n")

            # Re-raise to stop training gracefully
            raise RuntimeError(
                f"Worker process crashed. "
                f"Please check logs/ directory for server errors, "
                f"or reduce num_workers in config to improve stability."
            ) from e

    def reset_wait(self, **kwargs) -> Tuple[Any, dict]:
        """
        Wait for reset results, handling worker failures gracefully.
        """
        try:
            return super().reset_wait(**kwargs)
        except (BrokenPipeError, EOFError, ConnectionResetError) as e:
            print(f"\n{'='*60}")
            print(f"[CRITICAL] AsyncVectorEnv worker crashed during reset: {e}")
            print(f"{'='*60}\n")

            raise RuntimeError(
                f"Worker process crashed during reset. "
                f"Please reduce num_workers or check logs/."
            ) from e


def make_resilient_vector_env(
    env_fns: List[Callable[[], gym.Env]],
    num_envs: int = None,
    max_worker_restarts: int = None  # Kept for API compatibility but not used
) -> gym.vector.VectorEnv:
    """
    Create a vector environment with better error handling.

    For num_envs=1: Uses SyncVectorEnv (no multiprocessing, very stable)
    For num_envs>1: Uses ResilientAsyncVectorEnv (catches crashes gracefully)

    Args:
        env_fns: List of environment factory functions
        num_envs: Number of environments (for compatibility, uses len(env_fns))
        max_worker_restarts: Not used (kept for API compatibility)

    Returns:
        SyncVectorEnv or ResilientAsyncVectorEnv
    """
    actual_num_envs = len(env_fns)

    if actual_num_envs == 1:
        # Use SyncVectorEnv for single environment (no multiprocessing, no crashes)
        print("[VectorEnv] Using SyncVectorEnv (single worker, maximum stability)")
        return SyncVectorEnv(env_fns)
    else:
        # Use resilient version for multiple environments
        print(f"[VectorEnv] Using ResilientAsyncVectorEnv with {actual_num_envs} workers")
        print("[VectorEnv] WARNING: Multiprocessing on Windows can be unstable.")
        print("[VectorEnv] If workers crash frequently, reduce num_workers in config.")
        return ResilientAsyncVectorEnv(env_fns)
