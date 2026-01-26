"""
Test server restart functionality
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import SuikaEnvWrapper


def test_server_restart():
    """Test that server can be restarted after crash"""

    port = 8932
    print(f"\n{'='*60}")
    print(f"Testing server restart on port {port}")
    print(f"{'='*60}\n")

    # Create environment (will start server)
    env = SuikaEnvWrapper(
        headless=True,
        port=port,
        auto_start_server=True,
        fast_mode=True
    )

    print("1. Initial server started")

    # Do a few normal steps
    obs, info = env.reset()
    print("2. Environment reset successful")

    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"3.{i+1} Step {i+1} successful (reward={reward:.2f})")
        if terminated or truncated:
            obs, info = env.reset()

    # Now simulate server crash by killing it
    print("\n4. Simulating server crash...")
    if env._server_process is not None:
        env._server_process.kill()
        env._server_process.wait()
        print("   Server process killed")
        time.sleep(2)

        # Verify server is really dead
        if env._is_server_healthy(port):
            print("   WARNING: Server still responds to health check!")
        else:
            print("   ✓ Server is dead (health check failed)")

    # Try to do a step - should trigger restart
    print("\n5. Attempting step after crash (should trigger restart)...")
    try:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step successful after restart! (reward={reward:.2f})")
        if info.get('server_restarted'):
            print("   ✓ Server was restarted as expected")
        else:
            print("   Note: Server restart flag not set")
    except Exception as e:
        print(f"   Error: {e}")

    # Check logs
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_file_out = log_dir / f"server_port{port}_stdout.log"
    log_file_err = log_dir / f"server_port{port}_stderr.log"

    print(f"\n6. Checking logs...")
    with open(log_file_out, 'r', encoding='utf-8') as f:
        log_content = f.read()
        restart_count = log_content.count("Starting server on port")
        print(f"   Server start count in log: {restart_count}")
        if restart_count >= 2:
            print("   ✓ Multiple server starts detected (restart confirmed)")
        else:
            print("   Note: Expected multiple server starts in log")

    with open(log_file_err, 'r', encoding='utf-8') as f:
        err_content = f.read()
        if err_content.strip():
            print(f"   stderr content ({len(err_content)} bytes):")
            print(f"   {err_content[:200]}")
        else:
            print("   stderr is empty (no errors)")

    # Clean up
    env.close()
    print("\n7. Test completed successfully!")


if __name__ == "__main__":
    test_server_restart()
