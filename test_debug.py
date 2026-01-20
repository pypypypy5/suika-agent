"""
디버그 스크립트 - 실제 환경과 에이전트 생성 테스트
"""

import sys
import numpy as np
from gymnasium import spaces

# SimpleAgent import
from agents.simple_agent import SimpleAgent

# Mock observation space (실제 환경과 동일)
observation_space = spaces.Dict({
    'image': spaces.Box(0, 255, (128, 128, 4), dtype=np.uint8),
    'score': spaces.Box(0.0, 1e6, (1,), dtype=np.float32)
})

# Mock action space (실제 환경과 동일)
action_space = spaces.Box(0.0, 1.0, (1,), dtype=np.float32)

# Config
config = {
    'gamma': 0.99,
    'learning_rate': 0.0003,
    'num_discrete_actions': 11,
    'obs_key': 'image',
    'network': {
        'hidden_dims': [128]
    }
}

print("=" * 60)
print("디버그: SimpleAgent 초기화")
print("=" * 60)

print(f"\nObservation space: {observation_space}")
print(f"  - image shape: {observation_space.spaces['image'].shape}")
print(f"Action space: {action_space}")

# Agent 생성
agent = SimpleAgent(
    observation_space=observation_space,
    action_space=action_space,
    config=config,
    device='cpu'
)

print(f"\nAgent 생성 완료!")
print(f"  - is_dict_obs: {agent.is_dict_obs}")
print(f"  - obs_key: {agent.obs_key}")
print(f"  - obs_shape: {agent.obs_shape}")
print(f"  - action_dim: {agent.action_dim}")
print(f"  - is_discrete_env: {agent.is_discrete_env}")

# 네트워크 첫 번째 Conv2d 확인
first_conv = agent.policy_net.encoder[0]
print(f"\n첫 번째 Conv2d:")
print(f"  - Input channels: {first_conv.in_channels}")
print(f"  - Output channels: {first_conv.out_channels}")
print(f"  - Weight shape: {first_conv.weight.shape}")

# Mock observation 생성
mock_obs = {
    'image': np.random.randint(0, 255, (128, 128, 4), dtype=np.uint8),
    'score': np.array([0.0], dtype=np.float32)
}

print(f"\nMock observation:")
print(f"  - image shape: {mock_obs['image'].shape}")
print(f"  - image dtype: {mock_obs['image'].dtype}")

# Extract observation
extracted_obs = agent._extract_observation(mock_obs)
print(f"\nExtracted observation:")
print(f"  - shape: {extracted_obs.shape}")
print(f"  - dtype: {extracted_obs.dtype}")

# Preprocess observation
obs_tensor = agent.preprocess_observation(extracted_obs)
print(f"\nPreprocessed observation (tensor):")
print(f"  - shape: {obs_tensor.shape}")
print(f"  - dtype: {obs_tensor.dtype}")
print(f"  - device: {obs_tensor.device}")

# Action 선택
print(f"\n행동 선택 테스트...")
try:
    action = agent.select_action(mock_obs, deterministic=True)
    print(f"✓ 성공!")
    print(f"  - action: {action}")
    print(f"  - action type: {type(action)}")
    print(f"  - action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
except Exception as e:
    print(f"✗ 실패!")
    print(f"  - Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
