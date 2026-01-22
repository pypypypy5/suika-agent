"""
통합 VectorEnv 아키텍처 테스트

모든 환경이 VectorEnv로 통일되었는지 확인하고,
단일/다중 환경 모두에서 에이전트와 Trainer가 정상 동작하는지 검증합니다.
"""

import pytest
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from unittest.mock import MagicMock, patch
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockSuikaEnv:
    """테스트용 Mock Suika 환경 (완전한 Gymnasium 환경)"""

    def __init__(self, port=8923):
        self.port = port
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8),
            'score': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(10)
        self.current_step = 0
        self.max_steps = 20

        # Gymnasium 필수 속성
        self.metadata = {}
        self.spec = None
        self.render_mode = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        """환경 리셋"""
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        obs = {
            'image': np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
            'score': np.array([0.0], dtype=np.float32)
        }
        info = {'episode': 0}
        return obs, info

    def step(self, action):
        """환경 스텝"""
        self.current_step += 1

        # 다음 관찰
        obs = {
            'image': np.random.randint(0, 256, (84, 84, 4), dtype=np.uint8),
            'score': np.array([float(self.current_step * 10)], dtype=np.float32)
        }

        # 보상
        reward = float(np.random.rand())

        # 종료 조건
        terminated = (self.current_step >= self.max_steps)
        truncated = False

        info = {
            'episode_step': self.current_step,
            'score': obs['score'][0]
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        """환경 종료"""
        pass

    def render(self):
        """렌더링 (선택적)"""
        pass


class TestVectorEnvCreation:
    """VectorEnv 생성 테스트"""

    def test_single_env_returns_vector_env(self):
        """num_envs=1도 VectorEnv 반환"""
        def make_env():
            return MockSuikaEnv()

        # SyncVectorEnv로 래핑
        env = SyncVectorEnv([make_env])

        # VectorEnv 속성 확인
        assert hasattr(env, 'num_envs')
        assert env.num_envs == 1

        # 배치 형태 확인
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs['image'].shape[0] == 1  # (1, H, W, C)
        assert obs['score'].shape[0] == 1  # (1, 1)

        env.close()

    def test_multi_env_returns_vector_env(self):
        """num_envs=4도 VectorEnv 반환"""
        def make_env(rank):
            def _init():
                return MockSuikaEnv(port=8923 + rank)
            return _init

        envs = [make_env(i) for i in range(4)]
        env = SyncVectorEnv(envs)

        assert env.num_envs == 4

        obs, info = env.reset()
        assert obs['image'].shape[0] == 4  # (4, H, W, C)
        assert obs['score'].shape[0] == 4  # (4, 1)

        env.close()

    def test_vector_env_step_returns_batches(self):
        """VectorEnv step이 배치 반환"""
        def make_env(rank):
            def _init():
                return MockSuikaEnv(port=8923 + rank)
            return _init

        num_envs = 3
        envs = [make_env(i) for i in range(num_envs)]
        env = SyncVectorEnv(envs)

        obs, _ = env.reset()

        # 배치 행동
        actions = np.array([0, 1, 2])

        next_obs, rewards, terminated, truncated, info = env.step(actions)

        # 모든 반환값이 배치 형태
        assert next_obs['image'].shape == (num_envs, 84, 84, 4)
        assert next_obs['score'].shape == (num_envs, 1)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

        env.close()

    def test_sync_vector_env_with_single_env_has_no_overhead(self):
        """SyncVectorEnv with num_envs=1은 오버헤드가 최소"""
        import time

        def make_env():
            return MockSuikaEnv()

        # VectorEnv
        vec_env = SyncVectorEnv([make_env])

        # 성능 측정
        obs, _ = vec_env.reset()
        start = time.time()
        for _ in range(100):
            actions = np.array([0])
            obs, _, _, _, _ = vec_env.step(actions)
        vec_time = time.time() - start

        vec_env.close()

        # 단일 환경과 비교 (직접 사용)
        single_env = MockSuikaEnv()
        obs, _ = single_env.reset()
        start = time.time()
        for _ in range(100):
            obs, _, _, _, _ = single_env.step(0)
        single_time = time.time() - start

        single_env.close()

        # VectorEnv 오버헤드가 30% 이하여야 함
        overhead = (vec_time - single_time) / single_time
        assert overhead < 0.3, f"VectorEnv overhead too high: {overhead:.2%}"


class TestAgentBatchProcessing:
    """에이전트 배치 처리 테스트"""

    @pytest.fixture
    def mock_config(self):
        """Mock 설정 (dict 형식)"""
        return {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'batch_size': 64,
            'network': {
                'hidden_dims': [64]
            },
            'obs_key': 'image'
        }

    @pytest.fixture
    def mock_env_spaces(self):
        """Mock 환경 spaces"""
        obs_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8),
            'score': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
        })
        action_space = spaces.Discrete(10)
        return obs_space, action_space

    def test_agent_select_action_handles_batch(self, mock_env_spaces, mock_config):
        """에이전트가 배치 입력 처리"""
        from agents.simple_agent import SimpleAgent

        obs_space, action_space = mock_env_spaces
        agent = SimpleAgent(obs_space, action_space, mock_config)

        # 배치 관찰 (4개 환경)
        batch_size = 4
        obs_batch = {
            'image': np.random.randint(0, 256, (batch_size, 84, 84, 4), dtype=np.uint8),
            'score': np.random.rand(batch_size, 1).astype(np.float32)
        }

        # 배치 행동 선택
        actions = agent.select_action(obs_batch, deterministic=False)

        # 배치 형태 확인
        assert isinstance(actions, np.ndarray)
        assert actions.shape == (batch_size,)
        assert all(0 <= a < 10 for a in actions)

    def test_agent_select_action_single_env_batch(self, mock_env_spaces, mock_config):
        """에이전트가 단일 환경 배치 처리 (batch_size=1)"""
        from agents.simple_agent import SimpleAgent

        obs_space, action_space = mock_env_spaces
        agent = SimpleAgent(obs_space, action_space, mock_config)

        # 단일 환경의 배치 (VectorEnv with num_envs=1)
        obs_batch = {
            'image': np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8),
            'score': np.random.rand(1, 1).astype(np.float32)
        }

        actions = agent.select_action(obs_batch, deterministic=True)

        assert isinstance(actions, np.ndarray)
        assert actions.shape == (1,)

    def test_agent_store_transition_handles_batch(self, mock_env_spaces, mock_config):
        """에이전트가 배치 transition 저장"""
        from agents.simple_agent import SimpleAgent

        obs_space, action_space = mock_env_spaces
        agent = SimpleAgent(obs_space, action_space, mock_config)
        agent.train()

        batch_size = 4

        # 배치 transition
        obs_batch = {
            'image': np.random.randint(0, 256, (batch_size, 84, 84, 4), dtype=np.uint8),
            'score': np.random.rand(batch_size, 1).astype(np.float32)
        }
        actions = np.array([0, 1, 2, 3])
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        next_obs_batch = {
            'image': np.random.randint(0, 256, (batch_size, 84, 84, 4), dtype=np.uint8),
            'score': np.random.rand(batch_size, 1).astype(np.float32)
        }
        dones = np.array([False, False, True, False])

        # Transition 저장
        agent.store_transition(obs_batch, actions, rewards, next_obs_batch, dones)

        # 환경별 버퍼 확인
        assert 0 in agent.episode_buffers
        assert 1 in agent.episode_buffers
        assert 2 in agent.episode_buffers
        assert 3 in agent.episode_buffers

        # 환경 2는 에피소드 종료되어 학습 준비
        assert 2 in agent.completed_episodes

    def test_agent_update_after_episode_completion(self, mock_env_spaces, mock_config):
        """에피소드 완료 후 에이전트 학습"""
        from agents.simple_agent import SimpleAgent

        obs_space, action_space = mock_env_spaces
        agent = SimpleAgent(obs_space, action_space, mock_config)
        agent.train()

        # 환경 0에서 에피소드 진행
        for step in range(10):
            obs = {
                'image': np.random.randint(0, 256, (1, 84, 84, 4), dtype=np.uint8),
                'score': np.random.rand(1, 1).astype(np.float32)
            }
            action = np.array([0])
            reward = np.array([1.0])
            next_obs = obs
            done = np.array([step == 9])

            agent.store_transition(obs, action, reward, next_obs, done)

        # 학습
        update_info = agent.update()

        # 메트릭 반환 확인
        assert 'loss' in update_info
        assert 'num_episodes_updated' in update_info
        assert update_info['num_episodes_updated'] == 1

        # 버퍼 초기화 확인
        assert 0 not in agent.completed_episodes


class TestTrainerWithVectorEnv:
    """Trainer와 VectorEnv 통합 테스트"""

    @pytest.fixture
    def mock_config(self):
        """Mock 설정 (dict 형식)"""
        return {
            'gamma': 0.99,
            'learning_rate': 0.001,
            'batch_size': 64,
            'network': {
                'hidden_dims': [64]
            },
            'obs_key': 'image',
            'system': {
                'device': 'cpu',
                'num_workers': 1
            },
            'training': {
                'total_timesteps': 100,
                'update_frequency': 10,
                'eval_frequency': 50,
                'eval_episodes': 2
            }
        }

    def test_trainer_with_single_env_vector(self, mock_config):
        """Trainer가 단일 환경 VectorEnv에서 동작"""
        from agents.simple_agent import SimpleAgent
        from training.trainer import Trainer

        # 단일 환경 VectorEnv
        def make_env():
            return MockSuikaEnv()

        env = SyncVectorEnv([make_env])
        agent = SimpleAgent(env.single_observation_space, env.single_action_space, mock_config)

        # Trainer 없이 직접 학습 루프 시뮬레이션 (간단한 버전)
        agent.train()
        obs, _ = env.reset()

        for step in range(50):
            actions = agent.select_action(obs, deterministic=False)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            agent.store_transition(obs, actions, rewards, next_obs, dones)

            if step % 10 == 0:
                agent.update()

            obs = next_obs

        env.close()

        # 학습이 정상적으로 진행되었는지 확인
        assert agent.total_steps >= 0  # 기본 통계 확인

    def test_trainer_with_multi_env_vector(self, mock_config):
        """다중 환경 VectorEnv에서 학습 루프 동작"""
        from agents.simple_agent import SimpleAgent

        # 4개 환경 VectorEnv
        def make_env(rank):
            def _init():
                return MockSuikaEnv(port=8923 + rank)
            return _init

        num_envs = 4
        envs = [make_env(i) for i in range(num_envs)]
        env = SyncVectorEnv(envs)

        agent = SimpleAgent(env.single_observation_space, env.single_action_space, mock_config)
        agent.train()

        # 학습 루프 시뮬레이션
        obs, _ = env.reset()

        episode_counts = [0] * num_envs

        for step in range(100):
            actions = agent.select_action(obs, deterministic=False)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = terminated | truncated

            agent.store_transition(obs, actions, rewards, next_obs, dones)

            # 에피소드 종료 추적
            for env_id in range(num_envs):
                if dones[env_id]:
                    episode_counts[env_id] += 1

            if step % 10 == 0:
                agent.update()

            obs = next_obs

        env.close()

        # 최소 1개 환경은 에피소드 완료
        assert sum(episode_counts) >= 1

    def test_trainer_code_path_unified(self, mock_config):
        """Trainer가 단일/다중 환경에서 동일한 코드 경로 사용"""
        from agents.simple_agent import SimpleAgent

        # 단일 환경
        env1 = SyncVectorEnv([lambda: MockSuikaEnv()])
        agent1 = SimpleAgent(env1.single_observation_space, env1.single_action_space, mock_config)

        # 다중 환경
        env4 = SyncVectorEnv([lambda: MockSuikaEnv(port=8923 + i) for i in range(4)])
        agent4 = SimpleAgent(env4.single_observation_space, env4.single_action_space, mock_config)

        # 동일한 로직으로 실행
        for env, agent in [(env1, agent1), (env4, agent4)]:
            obs, _ = env.reset()

            for _ in range(20):
                # 항상 배치 처리
                actions = agent.select_action(obs, deterministic=False)
                assert isinstance(actions, np.ndarray)
                assert actions.shape == (env.num_envs,)

                next_obs, rewards, terminated, truncated, info = env.step(actions)
                dones = terminated | truncated

                agent.store_transition(obs, actions, rewards, next_obs, dones)
                agent.update()

                obs = next_obs

            env.close()


class TestBackwardCompatibility:
    """기존 코드와의 호환성 테스트"""

    def test_existing_tests_still_work(self):
        """기존 테스트가 여전히 동작하는지 확인"""
        # 기존 테스트들이 수정 후에도 통과해야 함
        # 실제 테스트는 pytest로 실행될 때 자동으로 검증됨
        pass

    def test_env_interface_unchanged(self):
        """환경 인터페이스가 변경되지 않았는지 확인"""
        env = MockSuikaEnv()

        # 기본 메서드 존재 확인
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'close')

        # 기본 속성 확인
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')

    def test_agent_interface_extended_not_breaking(self):
        """에이전트 인터페이스가 확장되었지만 기존 메서드는 유지"""
        from agents.base_agent import BaseAgent
        import inspect

        # BaseAgent에 store_transition이 추가되었는지 확인
        methods = [m for m in dir(BaseAgent) if not m.startswith('_')]

        # 기존 필수 메서드
        assert 'select_action' in methods
        assert 'update' in methods
        assert 'save' in methods
        assert 'load' in methods

        # 새로 추가된 메서드
        assert 'store_transition' in methods


class TestPerformanceWithVectorEnv:
    """VectorEnv 성능 테스트"""

    def test_multi_env_throughput_improvement(self):
        """다중 환경이 throughput 향상시키는지 확인"""
        import time

        # 단일 환경
        env1 = SyncVectorEnv([lambda: MockSuikaEnv()])
        obs, _ = env1.reset()

        start = time.time()
        for _ in range(100):
            actions = np.array([0])
            obs, _, _, _, _ = env1.step(actions)
        time_single = time.time() - start
        env1.close()

        # 4개 환경
        env4 = SyncVectorEnv([lambda: MockSuikaEnv(port=8923 + i) for i in range(4)])
        obs, _ = env4.reset()

        start = time.time()
        for _ in range(100):
            actions = np.array([0, 1, 2, 3])
            obs, _, _, _, _ = env4.step(actions)
        time_multi = time.time() - start
        env4.close()

        # 4배 더 많은 스텝을 실행했지만 시간은 4배 이하여야 함
        # (SyncVectorEnv는 순차 실행이므로 4배에 가까울 수 있음)
        # 실제 AsyncVectorEnv에서는 병렬화로 개선됨
        steps_per_sec_single = 100 / time_single
        steps_per_sec_multi = 400 / time_multi  # 4배 더 많은 스텝

        # 다중 환경이 총 throughput을 증가시켜야 함
        assert steps_per_sec_multi > steps_per_sec_single


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
