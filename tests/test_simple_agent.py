"""
Simple Agent 테스트

SimpleAgent의 기본 기능을 테스트합니다.
"""

import pytest
import numpy as np
import torch
from gymnasium import spaces

from agents.simple_agent import SimpleAgent, SimplePolicyNetwork


class TestSimplePolicyNetwork:
    """SimplePolicyNetwork 유닛 테스트"""

    def test_vector_input(self):
        """벡터 입력에 대한 네트워크 생성 및 forward"""
        obs_shape = (4,)
        action_dim = 2
        hidden_dim = 64

        network = SimplePolicyNetwork(obs_shape, action_dim, hidden_dim)

        # Forward pass
        batch_size = 8
        x = torch.randn(batch_size, *obs_shape)
        output = network(x)

        # 출력 shape 확인
        assert output.shape == (batch_size, action_dim)

    def test_image_input(self):
        """이미지 입력에 대한 네트워크 생성 및 forward"""
        obs_shape = (3, 84, 84)  # (C, H, W)
        action_dim = 5
        hidden_dim = 128

        network = SimplePolicyNetwork(obs_shape, action_dim, hidden_dim)

        # Forward pass
        batch_size = 4
        x = torch.randn(batch_size, *obs_shape)
        output = network(x)

        # 출력 shape 확인
        assert output.shape == (batch_size, action_dim)


class TestSimpleAgent:
    """SimpleAgent 유닛 테스트"""

    @pytest.fixture
    def vector_env_spaces(self):
        """벡터 환경 spaces"""
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        return obs_space, action_space

    @pytest.fixture
    def image_env_spaces(self):
        """이미지 환경 spaces"""
        obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        action_space = spaces.Discrete(5)
        return obs_space, action_space

    @pytest.fixture
    def config(self):
        """기본 설정"""
        return {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'network': {
                'hidden_dims': [128]
            }
        }

    def test_agent_initialization_vector(self, vector_env_spaces, config):
        """벡터 환경에서 에이전트 초기화"""
        obs_space, action_space = vector_env_spaces

        agent = SimpleAgent(obs_space, action_space, config, device='cpu')

        # 기본 속성 확인
        assert agent.observation_space == obs_space
        assert agent.action_space == action_space
        assert agent.action_dim == 2
        assert agent.is_discrete is True
        assert agent.policy_net is not None
        assert agent.optimizer is not None

    def test_agent_initialization_image(self, image_env_spaces, config):
        """이미지 환경에서 에이전트 초기화"""
        obs_space, action_space = image_env_spaces

        agent = SimpleAgent(obs_space, action_space, config, device='cpu')

        assert agent.action_dim == 5
        assert agent.policy_net is not None

    def test_select_action_deterministic(self, vector_env_spaces, config):
        """결정적 행동 선택"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        agent.eval()

        obs = obs_space.sample()
        action = agent.select_action(obs, deterministic=True)

        # 행동이 유효한 범위인지 확인
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 2

    def test_select_action_stochastic(self, vector_env_spaces, config):
        """확률적 행동 선택"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        agent.train()

        obs = obs_space.sample()

        # 여러 번 샘플링해서 확률적으로 다른 행동이 나오는지 확인
        actions = [agent.select_action(obs, deterministic=False) for _ in range(10)]

        # 모두 유효한 행동인지 확인
        for action in actions:
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < 2

    def test_store_transition(self, vector_env_spaces, config):
        """전이 저장"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        agent.train()

        # 행동 선택 (log_prob 저장됨)
        obs = obs_space.sample()
        agent.select_action(obs, deterministic=False)

        # 보상 저장
        reward = 1.0
        agent.store_transition(reward)

        assert len(agent.episode_rewards_buffer) == 1
        assert agent.episode_rewards_buffer[0] == reward

    def test_update_trainer_style(self, vector_env_spaces, config):
        """Trainer 스타일로 업데이트 (호환성 테스트)"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        agent.train()

        # Trainer처럼 에피소드 시뮬레이션
        episode_length = 10
        for step in range(episode_length):
            obs = obs_space.sample()
            action = agent.select_action(obs, deterministic=False)
            next_obs = obs_space.sample()
            reward = np.random.rand()
            done = (step == episode_length - 1)

            # Trainer가 호출하는 방식
            metrics = agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done
            )

            # 중간 스텝에서는 빈 딕셔너리 반환
            if not done:
                assert metrics == {}
            else:
                # 마지막 스텝에서만 메트릭 반환
                assert 'loss' in metrics
                assert 'episode_reward' in metrics
                assert 'episode_length' in metrics
                assert metrics['episode_length'] == episode_length

        # 버퍼가 초기화되었는지 확인
        assert len(agent.episode_log_probs) == 0
        assert len(agent.episode_rewards_buffer) == 0

    def test_full_episode(self, vector_env_spaces, config):
        """전체 에피소드 시뮬레이션 (Trainer 호환 방식)"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        agent.train()

        # 에피소드 실행
        total_reward = 0
        episode_length = 20

        for step in range(episode_length):
            obs = obs_space.sample()
            action = agent.select_action(obs, deterministic=False)
            next_obs = obs_space.sample()

            # 간단한 보상 함수
            reward = 1.0 if action == 0 else -1.0
            done = (step == episode_length - 1)

            # Trainer 스타일로 업데이트
            metrics = agent.update(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done
            )
            agent.total_steps += 1  # Trainer가 관리하는 방식 시뮬레이션
            total_reward += reward

            if done:
                agent.episodes += 1  # Trainer가 관리하는 방식 시뮬레이션

        assert metrics['episode_reward'] == total_reward
        assert metrics['episode_length'] == episode_length
        assert agent.episodes == 1
        assert agent.total_steps == episode_length

    def test_save_and_load(self, vector_env_spaces, config, tmp_path):
        """모델 저장 및 로드"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')

        # 일부 학습
        agent.train()
        for _ in range(5):
            obs = obs_space.sample()
            agent.select_action(obs, deterministic=False)
            agent.store_transition(1.0)
        agent.update()

        # 저장
        save_path = tmp_path / "agent_checkpoint.pt"
        agent.save(str(save_path))

        assert save_path.exists()

        # 새 에이전트 생성 및 로드
        new_agent = SimpleAgent(obs_space, action_space, config, device='cpu')
        new_agent.load(str(save_path))

        # 통계가 로드되었는지 확인
        assert new_agent.total_steps == agent.total_steps
        assert new_agent.episodes == agent.episodes

    def test_eval_mode(self, vector_env_spaces, config):
        """평가 모드 전환"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')

        # 평가 모드로 전환
        agent.eval()
        assert not agent.policy_net.training

        # 학습 모드로 전환
        agent.train()
        assert agent.policy_net.training

    def test_statistics(self, vector_env_spaces, config):
        """통계 조회"""
        obs_space, action_space = vector_env_spaces
        agent = SimpleAgent(obs_space, action_space, config, device='cpu')

        # 초기 통계
        stats = agent.get_statistics()
        assert 'total_steps' in stats
        assert 'episodes' in stats

        # 에피소드 실행 후 (Trainer 스타일)
        agent.train()
        for step in range(10):
            obs = obs_space.sample()
            action = agent.select_action(obs, deterministic=False)
            next_obs = obs_space.sample()
            reward = 1.0
            done = (step == 9)

            agent.update(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
            agent.total_steps += 1  # Trainer가 관리
            if done:
                agent.episodes += 1  # Trainer가 관리

        stats = agent.get_statistics()
        assert stats['total_steps'] == 10
        assert stats['episodes'] == 1
        assert 'mean_episode_reward' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
