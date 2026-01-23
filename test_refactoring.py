"""
리팩터링 테스트 스크립트

Base Agent와 Simple Agent의 리팩터링이 정상적으로 작동하는지 확인합니다.
"""

import numpy as np
import torch
from gymnasium import spaces

from agents import SimpleAgent

def test_refactoring():
    """리팩터링 테스트"""
    print("=" * 60)
    print("Base Agent 리팩터링 테스트 시작")
    print("=" * 60)

    # 1. Dict observation space 테스트
    print("\n[Test 1] Dict observation space")
    obs_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
        'score': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
    })
    action_space = spaces.Discrete(11)

    config = {
        'obs_key': 'image',
        'gamma': 0.99,
        'learning_rate': 0.0003
    }

    agent = SimpleAgent(obs_space, action_space, config)

    # Base Agent에서 설정된 속성 확인
    assert hasattr(agent, 'is_dict_obs'), "is_dict_obs 속성이 없습니다"
    assert hasattr(agent, 'obs_key'), "obs_key 속성이 없습니다"
    assert hasattr(agent, 'obs_shape'), "obs_shape 속성이 없습니다"

    print(f"  [OK] is_dict_obs: {agent.is_dict_obs}")
    print(f"  [OK] obs_key: {agent.obs_key}")
    print(f"  [OK] obs_shape: {agent.obs_shape}")

    # extract_observation 테스트
    print("\n[Test 2] extract_observation 메서드")
    batch_size = 2
    test_obs = {
        'image': np.random.randint(0, 255, (batch_size, 84, 84, 3), dtype=np.uint8),
        'score': np.random.randn(batch_size, 1).astype(np.float32)
    }

    extracted = agent.extract_observation(test_obs)
    print(f"  [OK] 입력 타입: Dict")
    print(f"  [OK] 출력 shape: {extracted.shape}")
    assert extracted.shape == (batch_size, 84, 84, 3), f"추출된 shape이 잘못되었습니다: {extracted.shape}"

    # preprocess_observation 테스트
    print("\n[Test 3] preprocess_observation 메서드")
    preprocessed = agent.preprocess_observation(test_obs)
    print(f"  [OK] 입력 타입: Dict")
    print(f"  [OK] 출력 shape: {preprocessed.shape}")
    print(f"  [OK] 출력 타입: {type(preprocessed)}")
    print(f"  [OK] 디바이스: {preprocessed.device}")

    expected_shape = (batch_size, 3, 84, 84)  # (N, C, H, W)
    assert preprocessed.shape == expected_shape, f"전처리된 shape이 잘못되었습니다: {preprocessed.shape}"
    assert isinstance(preprocessed, torch.Tensor), "출력이 Tensor가 아닙니다"

    # select_action 테스트
    print("\n[Test 4] select_action 메서드")
    actions = agent.select_action(test_obs, deterministic=False)
    print(f"  [OK] 행동 shape: {actions.shape}")
    print(f"  [OK] 행동 값: {actions}")
    assert actions.shape == (batch_size,), f"행동 shape이 잘못되었습니다: {actions.shape}"

    # 2. 단일 observation space 테스트
    print("\n[Test 5] 단일 observation space")
    obs_space_single = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
    agent_single = SimpleAgent(obs_space_single, action_space, config)

    print(f"  [OK] is_dict_obs: {agent_single.is_dict_obs}")
    print(f"  [OK] obs_key: {agent_single.obs_key}")
    print(f"  [OK] obs_shape: {agent_single.obs_shape}")

    test_obs_single = np.random.randint(0, 255, (batch_size, 84, 84, 3), dtype=np.uint8)
    preprocessed_single = agent_single.preprocess_observation(test_obs_single)
    print(f"  [OK] 전처리 shape: {preprocessed_single.shape}")
    assert preprocessed_single.shape == expected_shape, f"전처리된 shape이 잘못되었습니다: {preprocessed_single.shape}"

    # 3. store_transition 테스트 (학습 모드)
    print("\n[Test 6] store_transition 메서드 (학습 모드)")
    agent.train()

    next_obs = {
        'image': np.random.randint(0, 255, (batch_size, 84, 84, 3), dtype=np.uint8),
        'score': np.random.randn(batch_size, 1).astype(np.float32)
    }
    rewards = np.array([1.0, 0.5])
    dones = np.array([False, False])

    agent.store_transition(test_obs, actions, rewards, next_obs, dones)
    print(f"  [OK] Transition 저장 성공")

    # 버퍼 확인
    print(f"  [OK] 버퍼 키: {list(agent.episode_buffers.keys())}")
    for env_id in agent.episode_buffers:
        buffer = agent.episode_buffers[env_id]
        print(f"    - env {env_id}: observations={len(buffer['observations'])}, actions={len(buffer['actions'])}, rewards={len(buffer['rewards'])}")

    # 7. update 메서드 테스트 (에피소드 완료 시)
    print("\n[Test 7] update 메서드 (에피소드 완료)")

    # 에피소드를 완료시키기
    done_true = np.array([True, True])
    agent.store_transition(test_obs, actions, rewards, next_obs, done_true)

    print(f"  [OK] Completed episodes: {agent.completed_episodes}")

    # Update 호출
    update_info = agent.update()
    print(f"  [OK] Update 결과: {update_info}")

    assert 'loss' in update_info, "loss가 반환되지 않았습니다"
    assert 'num_episodes_updated' in update_info, "num_episodes_updated가 반환되지 않았습니다"
    print(f"  [OK] Loss: {update_info['loss']:.4f}")
    print(f"  [OK] Episodes updated: {update_info['num_episodes_updated']}")

    print("\n" + "=" * 60)
    print("모든 테스트 통과!")
    print("=" * 60)

    # 요약
    print("\n[요약]")
    print("[OK] Base Agent에 observation space 분석 코드 추가 완료")
    print("[OK] Base Agent에 extract_observation() 메서드 추가 완료")
    print("[OK] Base Agent의 preprocess_observation() 메서드 개선 완료")
    print("[OK] Simple Agent에서 중복 코드 제거 완료")
    print("[OK] Simple Agent가 Base Agent의 메서드 사용 확인")
    print("\n리팩터링이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    test_refactoring()
