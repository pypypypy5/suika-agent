"""
Test worker stability with exception handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from envs.suika_wrapper import SuikaEnvWrapper


def test_reset_exception_handling():
    """
    Test that reset() doesn't crash worker when base env raises exception.
    """
    # Create wrapper with mock env
    wrapper = SuikaEnvWrapper(use_mock=True, normalize_obs=True)

    # Patch the base env's reset to raise an exception
    original_reset = wrapper.env.reset

    def failing_reset(*args, **kwargs):
        raise RuntimeError("Mock server error")

    wrapper.env.reset = failing_reset

    # This should NOT crash, but return dummy observation
    obs, info = wrapper.reset()

    # Check that we got a valid observation
    assert obs is not None
    assert 'image' in obs
    assert 'score' in obs
    assert info.get('forced_reset') == True
    assert 'error' in info

    # Restore original reset
    wrapper.env.reset = original_reset
    wrapper.close()

    print("✓ reset() exception handling test passed")


def test_step_exception_handling():
    """
    Test that step() doesn't crash worker when base env raises exception.
    """
    # Create wrapper with mock env
    wrapper = SuikaEnvWrapper(use_mock=True, normalize_obs=True)

    # Reset first
    obs, info = wrapper.reset()

    # Patch the base env's step to raise an exception
    original_step = wrapper.env.step

    def failing_step(*args, **kwargs):
        raise RuntimeError("Mock server error")

    wrapper.env.step = failing_step

    # This should NOT crash, but return dummy observation + terminated=True
    action = wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)

    # Check that we got a valid observation
    assert obs is not None
    assert 'image' in obs
    assert 'score' in obs
    assert terminated == True  # Episode should be terminated
    assert info.get('forced_termination') == True
    assert 'error' in info
    assert reward == 0.0

    # Restore original step
    wrapper.env.step = original_step
    wrapper.close()

    print("✓ step() exception handling test passed")


def test_normal_operation_still_works():
    """
    Test that normal operation is not affected by exception handling.
    """
    # Create wrapper with mock env
    wrapper = SuikaEnvWrapper(use_mock=True, normalize_obs=True)

    # Reset
    obs, info = wrapper.reset()
    assert obs is not None
    assert 'forced_reset' not in info  # Normal reset

    # Step
    action = wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert obs is not None
    assert 'forced_termination' not in info  # Normal step

    wrapper.close()

    print("✓ Normal operation test passed")


if __name__ == '__main__':
    print("Testing worker stability with exception handling...")
    print("="*60)

    test_reset_exception_handling()
    test_step_exception_handling()
    test_normal_operation_still_works()

    print("="*60)
    print("All tests passed! Worker crash prevention is working.")
