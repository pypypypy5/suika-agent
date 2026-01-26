"""
Test that server logs are properly captured to files
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import SuikaEnvWrapper


def test_server_logging():
    """Test that server stdout/stderr are logged to files"""

    # Create environment (will start server)
    env = SuikaEnvWrapper(
        headless=True,
        port=8930,  # Use unique port for testing
        auto_start_server=True,
        fast_mode=True
    )

    # Check that log files were created
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_file_out = log_dir / "server_port8930_stdout.log"
    log_file_err = log_dir / "server_port8930_stderr.log"

    print(f"Checking for log files:")
    print(f"  stdout: {log_file_out}")
    print(f"  stderr: {log_file_err}")

    # Wait a bit for server to start and write logs
    time.sleep(2)

    # Check files exist
    assert log_file_out.exists(), f"stdout log file not found: {log_file_out}"
    assert log_file_err.exists(), f"stderr log file not found: {log_file_err}"

    # Check that stdout log has content (server startup messages)
    with open(log_file_out, 'r', encoding='utf-8') as f:
        stdout_content = f.read()
        print(f"\nstdout log content ({len(stdout_content)} bytes):")
        print(stdout_content[:500] if len(stdout_content) > 500 else stdout_content)
        assert len(stdout_content) > 0, "stdout log is empty"

    # Run a few steps to generate more logs
    obs, info = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    # Clean up
    env.close()

    # Check logs again after close
    with open(log_file_out, 'r', encoding='utf-8') as f:
        final_stdout = f.read()
        print(f"\nFinal stdout log ({len(final_stdout)} bytes)")

    with open(log_file_err, 'r', encoding='utf-8') as f:
        stderr_content = f.read()
        print(f"\nstderr log ({len(stderr_content)} bytes):")
        if stderr_content:
            print(stderr_content[:500] if len(stderr_content) > 500 else stderr_content)
        else:
            print("(empty - no errors)")

    print("\nServer logging test passed!")


if __name__ == "__main__":
    test_server_logging()
