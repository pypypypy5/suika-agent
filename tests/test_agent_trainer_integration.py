"""
Agent-Trainer 통합 테스트

BaseAgent를 상속한 모든 에이전트가 Trainer와 호환되는지 확인합니다.
"""

import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import MagicMock

from agents import RandomAgent, SimpleAgent
from agents.base_agent import BaseAgent


class MockEnv:
    """테스트용 Mock 환경"""

    def __init__(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = np.random.rand()
        terminated = np.random.rand() > 0.9  # 10% 확률로 종료
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class TestAgentTrainerInterface:
    """에이전트와 Trainer 간 인터페이스 호환성 테스트"""

    @pytest.fixture
    def mock_env(self):
        """Mock 환경"""
        return MockEnv()

    @pytest.fixture
    def config(self):
        """기본 설정"""
        return {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'network': {
                'hidden_dims': [64]
            }
        }

    def simulate_trainer_loop(self, agent: BaseAgent, env: MockEnv, num_steps: int = 50):
        """
        Trainer의 학습 루프를 시뮬레이션

        Args:
            agent: 테스트할 에이전트
            env: 환경
            num_steps: 시뮬레이션 스텝 수

        Returns:
            학습이 정상적으로 완료되었는지 여부
        """
        agent.train()
        obs, info = env.reset()

        episode_count = 0
        step_count = 0

        for step in range(num_steps):
            # 행동 선택 (Trainer의 line 85)
            action = agent.select_action(obs, deterministic=False)

            # 환경 스텝 (Trainer의 line 88)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 에이전트 업데이트 (Trainer의 line 93-99)
            try:
                update_info = agent.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done
                )

                # update_info가 딕셔너리여야 함
                assert isinstance(update_info, dict), \
                    f"update() must return dict, got {type(update_info)}"

            except Exception as e:
                pytest.fail(f"Agent update failed: {e}")

            # 통계 업데이트 (Trainer의 line 104)
            agent.total_steps += 1
            step_count += 1

            # 에피소드 종료 (Trainer의 line 107-142)
            if done:
                agent.episodes += 1
                episode_count += 1
                obs, info = env.reset()
            else:
                obs = next_obs

        return episode_count > 0 and step_count == num_steps

    def test_random_agent_trainer_compatibility(self, mock_env, config):
        """RandomAgent와 Trainer 호환성"""
        agent = RandomAgent(
            observation_space=mock_env.observation_space,
            action_space=mock_env.action_space,
            config=config
        )

        # 학습 루프 시뮬레이션
        success = self.simulate_trainer_loop(agent, mock_env, num_steps=50)
        assert success

    def test_simple_agent_trainer_compatibility(self, mock_env, config):
        """SimpleAgent와 Trainer 호환성"""
        agent = SimpleAgent(
            observation_space=mock_env.observation_space,
            action_space=mock_env.action_space,
            config=config,
            device='cpu'
        )

        # 학습 루프 시뮬레이션
        success = self.simulate_trainer_loop(agent, mock_env, num_steps=50)
        assert success

        # SimpleAgent는 학습을 해야 하므로 메트릭이 기록되어야 함
        assert len(agent.episode_rewards) > 0

    def test_agent_required_methods(self, mock_env, config):
        """모든 에이전트가 필수 메서드를 구현했는지 확인"""
        agents = [
            RandomAgent(mock_env.observation_space, mock_env.action_space, config),
            SimpleAgent(mock_env.observation_space, mock_env.action_space, config, device='cpu')
        ]

        required_methods = [
            'select_action',
            'update',
            'save',
            'load',
            'train',
            'eval'
        ]

        for agent in agents:
            for method_name in required_methods:
                assert hasattr(agent, method_name), \
                    f"{agent.__class__.__name__} missing method: {method_name}"

                method = getattr(agent, method_name)
                assert callable(method), \
                    f"{agent.__class__.__name__}.{method_name} is not callable"

    def test_agent_update_signature(self, mock_env, config):
        """에이전트의 update() 시그니처가 Trainer와 호환되는지 확인"""
        agents = [
            RandomAgent(mock_env.observation_space, mock_env.action_space, config),
            SimpleAgent(mock_env.observation_space, mock_env.action_space, config, device='cpu')
        ]

        obs = mock_env.observation_space.sample()
        action = mock_env.action_space.sample()
        next_obs = mock_env.observation_space.sample()
        reward = 1.0
        done = False

        for agent in agents:
            # Trainer가 호출하는 방식으로 테스트
            try:
                result = agent.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    done=done
                )

                assert isinstance(result, dict), \
                    f"{agent.__class__.__name__}.update() must return dict"

            except TypeError as e:
                pytest.fail(
                    f"{agent.__class__.__name__}.update() signature incompatible with Trainer: {e}"
                )

    def test_agent_select_action_signature(self, mock_env, config):
        """에이전트의 select_action() 시그니처 확인"""
        agents = [
            RandomAgent(mock_env.observation_space, mock_env.action_space, config),
            SimpleAgent(mock_env.observation_space, mock_env.action_space, config, device='cpu')
        ]

        obs = mock_env.observation_space.sample()

        for agent in agents:
            # 확률적 행동 선택
            action1 = agent.select_action(obs, deterministic=False)
            assert action1 is not None

            # 결정적 행동 선택
            action2 = agent.select_action(obs, deterministic=True)
            assert action2 is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
