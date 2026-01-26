"""
학습 재개 기능 테스트

체크포인트 저장 및 로드가 올바르게 작동하는지 테스트합니다.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import DQNAgent, SimpleAgent
from gymnasium import spaces


def test_dqn_save_load():
    """DQN 에이전트 저장 및 로드 테스트"""
    print("=" * 60)
    print("Testing DQN Agent Save/Load")
    print("=" * 60)

    # 1. 에이전트 생성
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
        'score': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32)
    })
    action_space = spaces.Discrete(11)

    config = {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epsilon_start': 1.0,
        'epsilon_min': 0.1,
        'epsilon_decay': 0.995,
        'buffer_capacity': 1000,
        'target_update_freq': 100,
        'obs_key': 'image',
        'network': {'hidden_dims': [128]}
    }

    agent1 = DQNAgent(obs_space, action_space, config)

    # 2. 에이전트 상태 변경 (학습 시뮬레이션)
    agent1.total_steps = 5000
    agent1.episodes = 100
    agent1.epsilon = 0.5

    print(f"Original Agent - Steps: {agent1.total_steps}, Episodes: {agent1.episodes}, Epsilon: {agent1.epsilon:.3f}")

    # 3. 임시 파일에 저장
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name

    agent1.save(tmp_path)
    print(f"\nSaved to: {tmp_path}")

    # 4. 새로운 에이전트 생성 및 로드
    agent2 = DQNAgent(obs_space, action_space, config)
    print(f"\nNew Agent before load - Steps: {agent2.total_steps}, Episodes: {agent2.episodes}, Epsilon: {agent2.epsilon:.3f}")

    agent2.load(tmp_path)
    print(f"New Agent after load - Steps: {agent2.total_steps}, Episodes: {agent2.episodes}, Epsilon: {agent2.epsilon:.3f}")

    # 5. 검증
    assert agent2.total_steps == 5000, f"Expected steps=5000, got {agent2.total_steps}"
    assert agent2.episodes == 100, f"Expected episodes=100, got {agent2.episodes}"
    assert abs(agent2.epsilon - 0.5) < 0.001, f"Expected epsilon=0.5, got {agent2.epsilon}"

    print("\n✅ DQN Save/Load Test Passed!")

    # 임시 파일 삭제
    Path(tmp_path).unlink()


def test_simple_agent_save_load():
    """SimpleAgent 저장 및 로드 테스트"""
    print("\n" + "=" * 60)
    print("Testing SimpleAgent Save/Load")
    print("=" * 60)

    # 1. 에이전트 생성
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
        'score': spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32)
    })
    action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    config = {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'num_discrete_actions': 11,
        'obs_key': 'image'
    }

    agent1 = SimpleAgent(obs_space, action_space, config)

    # 2. 에이전트 상태 변경
    agent1.total_steps = 3000
    agent1.episodes = 50

    print(f"Original Agent - Steps: {agent1.total_steps}, Episodes: {agent1.episodes}")

    # 3. 임시 파일에 저장
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        tmp_path = tmp.name

    agent1.save(tmp_path)
    print(f"\nSaved to: {tmp_path}")

    # 4. 새로운 에이전트 생성 및 로드
    agent2 = SimpleAgent(obs_space, action_space, config)
    print(f"\nNew Agent before load - Steps: {agent2.total_steps}, Episodes: {agent2.episodes}")

    agent2.load(tmp_path)
    print(f"New Agent after load - Steps: {agent2.total_steps}, Episodes: {agent2.episodes}")

    # 5. 검증
    assert agent2.total_steps == 3000, f"Expected steps=3000, got {agent2.total_steps}"
    assert agent2.episodes == 50, f"Expected episodes=50, got {agent2.episodes}"

    print("\n✅ SimpleAgent Save/Load Test Passed!")

    # 임시 파일 삭제
    Path(tmp_path).unlink()


if __name__ == "__main__":
    test_dqn_save_load()
    test_simple_agent_save_load()

    print("\n" + "=" * 60)
    print("All Tests Passed! 🎉")
    print("=" * 60)
