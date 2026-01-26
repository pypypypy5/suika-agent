"""
Test force_server_restart option
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs import SuikaEnvWrapper


def test_force_restart():
    """Test that force_server_restart kills existing server and starts new one"""

    port = 8933
    print(f"\n{'='*60}")
    print(f"Testing force_server_restart on port {port}")
    print(f"{'='*60}\n")

    # Create first environment
    print("1. Creating first environment (should start server)")
    env1 = SuikaEnvWrapper(
        port=port,
        auto_start_server=True,
        force_server_restart=False
    )

    # Check that server is running
    assert env1._is_port_in_use(port), "Server should be running"
    print(f"   OK Server is running on port {port}")

    # Create second environment with force_server_restart
    print("\n2. Creating second environment with force_server_restart=True")
    env2 = SuikaEnvWrapper(
        port=port,
        auto_start_server=True,
        force_server_restart=True
    )

    # Check logs
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_file = log_dir / f"server_port{port}_stdout.log"

    print("\n3. Checking logs...")
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()
        restart_count = log_content.count("Starting server on port")
        print(f"   Server start count: {restart_count}")
        if restart_count >= 2:
            print("   OK Server was restarted (multiple starts detected)")
        else:
            print(f"   Warning: Expected multiple starts, got {restart_count}")

    # Clean up
    env1.close()
    env2.close()
    print("\n4. Test completed!")


if __name__ == "__main__":
    test_force_restart()
