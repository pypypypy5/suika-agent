"""
Test device detection
"""
import yaml
from agents import DQNAgent
from gymnasium.spaces import Box, Dict as DictSpace
import numpy as np

# Load config
with open('config/default.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Create dummy spaces
obs_space = DictSpace({
    'image': Box(0, 1, (260, 384, 3), np.float32),
    'score': Box(0, 1e6, (1,), np.float32)
})
act_space = Box(0, 1, (1,), np.float32)

# Create agent
system_config = config.get('system', {})
device_setting = system_config.get('device', 'auto')
device = None if device_setting == 'auto' else device_setting

print(f"Config device setting: {device_setting}")
print(f"Device parameter: {device}")
print()

agent = DQNAgent(
    observation_space=obs_space,
    action_space=act_space,
    config=config.get('agent', {}),
    device=device
)

print(f"\nAgent device: {agent.device}")
