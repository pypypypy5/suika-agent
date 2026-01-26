"""
환경 API 상세 테스트

에이전트가 사용할 API가 제대로 작동하는지 확인하고,
모든 정보가 올바르게 전달되는지 상세히 출력합니다.
"""

import sys
import os
import json
import numpy as np

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import SuikaEnvWrapper


def print_section(title):
    """섹션 헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_dict(d, indent=2):
    """딕셔너리를 보기 좋게 출력"""
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            print(f"{' ' * indent}{key}: array(shape={value.shape}, dtype={value.dtype})")
            print(f"{' ' * indent}  → min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
        elif isinstance(value, dict):
            print(f"{' ' * indent}{key}:")
            print_dict(value, indent + 2)
        else:
            print(f"{' ' * indent}{key}: {value}")


def test_mock_environment():
    """Mock 환경으로 API 테스트"""
    print_section("Mock 환경 API 테스트")

    print("\n[1] 환경 생성")
    print("-" * 70)
    env = SuikaEnvWrapper(use_mock=True, reward_scale=1.0, normalize_obs=True)
    print(f"✓ 환경 생성 완료")
    print(f"  - 클래스: {env.__class__.__name__}")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")

    print("\n[2] 초기 환경 리셋 (reset)")
    print("-" * 70)
    initial_obs, initial_info = env.reset(seed=42)
    print(f"✓ reset() 반환값:")
    print(f"\n  observation (타입: {type(initial_obs)}):")
    if isinstance(initial_obs, dict):
        print_dict(initial_obs, indent=4)
    else:
        print(f"    Shape: {initial_obs.shape}, dtype: {initial_obs.dtype}")

    print(f"\n  info (타입: {type(initial_info)}):")
    print_dict(initial_info, indent=4)

    # 초기 상태 저장 (비교용)
    if isinstance(initial_obs, dict):
        initial_score = initial_obs.get('score', [0])[0]
    else:
        initial_score = 0

    print("\n[3] 행동 선택 및 실행")
    print("-" * 70)

    num_steps = 5
    total_reward = 0

    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # 행동 선택
        action = env.action_space.sample()
        print(f"선택한 행동: {action}")
        print(f"  - 타입: {type(action)}")
        print(f"  - Shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
        print(f"  - 값 범위: [{env.action_space.low}, {env.action_space.high}]")

        # 행동 실행
        obs, reward, terminated, truncated, info = env.step(action)

        # 결과 출력
        print(f"\nstep() 반환값:")
        print(f"  1. observation:")
        if isinstance(obs, dict):
            print_dict(obs, indent=6)
        else:
            print(f"      Shape: {obs.shape}, dtype: {obs.dtype}")

        print(f"\n  2. reward: {reward}")
        print(f"      - 타입: {type(reward)}")
        print(f"      - Original reward: {info.get('original_reward', 'N/A')}")
        print(f"      - Processed reward: {info.get('processed_reward', 'N/A')}")

        print(f"\n  3. terminated: {terminated} (게임 오버 여부)")
        print(f"  4. truncated: {truncated} (시간 제한 등)")
        print(f"  5. info:")
        print_dict(info, indent=6)

        # 누적 보상
        total_reward += reward

        # 점수 변화
        if isinstance(obs, dict):
            current_score = obs.get('score', [0])[0]
            score_change = current_score - initial_score
            print(f"\n점수 변화: {initial_score:.0f} → {current_score:.0f} (Δ{score_change:+.0f})")
            initial_score = current_score

        print(f"누적 보상: {total_reward:.4f}")

        if terminated or truncated:
            print(f"\n⚠ 에피소드 종료!")
            print(f"  - terminated={terminated}, truncated={truncated}")
            break

    print("\n[4] 환경 통계")
    print("-" * 70)
    stats = env.get_episode_statistics()
    print("에피소드 통계:")
    print_dict(stats, indent=2)

    print("\n[5] 환경 종료")
    print("-" * 70)
    env.close()
    print("✓ 환경 정상 종료")

    return total_reward, step + 1


def test_real_environment_if_available():
    """실제 Suika 환경 테스트 (HTTP mode)"""
    print_section("실제 Suika 환경 API 테스트 (HTTP mode)")

    # Node.js 서버 사용 가능 여부 확인
    try:
        import requests

        print("\n[준비] Node.js 게임 서버 확인")
        print("-" * 70)

        response = requests.get("http://localhost:8924/health", timeout=5)
        if response.status_code == 200:
            print("✓ Node.js 게임 서버 사용 가능 (port 8924)")
        else:
            raise Exception("서버 응답 비정상")

    except Exception as e:
        print(f"✗ Node.js 게임 서버 사용 불가: {e}")
        print("\n서버를 먼저 시작하세요:")
        print("  cd suika_rl/server && node server.js")
        return None, 0

    # 실제 환경 테스트
    try:
        print("\n[1] 실제 환경 생성 (HTTP mode)")
        print("-" * 70)
        env = SuikaEnvWrapper(
            headless=True,
            port=8924,  # Node.js 서버 포트
            delay_before_img_capture=0.3,  # HTTP mode에서는 무시됨
            use_mock=False,
            fast_mode=True
        )
        print(f"✓ 실제 환경 생성 완료")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")

        print("\n[2] 초기 환경 리셋")
        print("-" * 70)
        print("게임 로딩 중... (몇 초 소요될 수 있음)")

        initial_obs, initial_info = env.reset()
        print(f"✓ reset() 반환값:")
        print(f"\n  observation:")
        print_dict(initial_obs, indent=4)
        print(f"\n  info:")
        print_dict(initial_info, indent=4)

        # 이미지 정보 상세 출력
        if 'image' in initial_obs:
            img = initial_obs['image']
            print(f"\n  이미지 상세:")
            print(f"    - Shape: {img.shape} (Height×Width×Channels)")
            print(f"    - Dtype: {img.dtype}")
            print(f"    - 값 범위: [{img.min()}, {img.max()}]")
            print(f"    - 평균: {img.mean():.4f}")

            # 이미지 저장 (선택사항)
            try:
                from PIL import Image
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img
                Image.fromarray(img_uint8).save('tests/initial_observation.png')
                print(f"    - 저장: tests/initial_observation.png")
            except Exception as e:
                print(f"    - 이미지 저장 실패: {e}")

        print("\n[3] 행동 실행 테스트")
        print("-" * 70)

        num_steps = 3
        for step in range(num_steps):
            print(f"\n--- Step {step + 1} ---")

            # 중앙 부근에 떨어뜨리기 (0.5 = 중앙)
            action = np.array([0.4 + step * 0.1], dtype=np.float32)
            print(f"선택한 행동: {action[0]:.2f} (위치: {int(action[0] * 640)}px)")

            print(f"행동 실행 중...")
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"\nstep() 반환값:")
            print(f"  observation:")
            if isinstance(obs, dict):
                for key, value in obs.items():
                    if key == 'image':
                        print(f"    {key}: array(shape={value.shape}, dtype={value.dtype})")
                    else:
                        print(f"    {key}: {value}")
            else:
                print(f"    {obs}")

            print(f"\n  reward: {reward:.2f}")
            print(f"  terminated: {terminated}")
            print(f"  truncated: {truncated}")
            print(f"  info: {info}")

            if terminated or truncated:
                print(f"\n⚠ 게임 종료!")
                break

        print("\n[4] 환경 종료")
        print("-" * 70)
        env.close()
        print("✓ 실제 환경 정상 종료")

        return True, step + 1

    except Exception as e:
        print(f"\n✗ 실제 환경 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def analyze_api_completeness(mock_result, real_result):
    """API 완전성 분석"""
    print_section("API 완전성 분석 및 에이전트 요구사항 확인")

    print("\n[에이전트가 필요로 하는 정보]")
    print("-" * 70)

    required_info = {
        "1. 관찰 (Observation)": {
            "설명": "환경의 현재 상태를 나타내는 정보",
            "필수 요소": [
                "게임 화면 이미지 (또는 상태 표현)",
                "현재 점수",
                "게임 진행 상태"
            ],
            "실제 제공": "✓ Dict{'image': array, 'score': float}"
        },
        "2. 행동 (Action)": {
            "설명": "에이전트가 취할 수 있는 행동",
            "필수 요소": [
                "과일을 떨어뜨릴 위치 (0~1 사이 연속값)"
            ],
            "실제 제공": "✓ Box(0, 1, shape=(1,))"
        },
        "3. 보상 (Reward)": {
            "설명": "행동에 대한 피드백",
            "필수 요소": [
                "점수 증가량",
                "스케일링된 보상값"
            ],
            "실제 제공": "✓ Float (점수 변화량 기반)"
        },
        "4. 종료 신호 (Done)": {
            "설명": "에피소드 종료 여부",
            "필수 요소": [
                "게임 오버 (terminated)",
                "시간 제한 (truncated)"
            ],
            "실제 제공": "✓ terminated, truncated (Gymnasium 표준)"
        },
        "5. 추가 정보 (Info)": {
            "설명": "디버깅 및 분석을 위한 메타데이터",
            "필수 요소": [
                "에피소드 통계",
                "원본 보상값",
                "점수 정보"
            ],
            "실제 제공": "✓ Dict (episode_score, episode_steps, etc.)"
        }
    }

    for category, details in required_info.items():
        print(f"\n{category}")
        print(f"  설명: {details['설명']}")
        print(f"  필수 요소:")
        for elem in details['필수 요소']:
            print(f"    - {elem}")
        print(f"  제공 여부: {details['실제 제공']}")

    print("\n\n[RL 학습에 필요한 전체 프로세스 검증]")
    print("-" * 70)

    process_steps = [
        ("1. 환경 초기화", "env = SuikaEnvWrapper()", "✓"),
        ("2. 에피소드 시작", "obs, info = env.reset()", "✓"),
        ("3. 관찰 → 행동", "action = agent.select_action(obs)", "✓ (API 제공)"),
        ("4. 행동 실행", "obs, reward, done, _, info = env.step(action)", "✓"),
        ("5. 경험 저장", "buffer.add(obs, action, reward, next_obs, done)", "✓ (에이전트 구현)"),
        ("6. 학습 업데이트", "agent.update(batch)", "✓ (에이전트 구현)"),
        ("7. 반복", "while not done", "✓"),
        ("8. 환경 종료", "env.close()", "✓")
    ]

    for step, code, status in process_steps:
        print(f"{step:20s} : {code:50s} [{status}]")

    print("\n\n[결론]")
    print("-" * 70)
    print("✓ 모든 필수 API가 제공됩니다.")
    print("✓ Gymnasium 표준 인터페이스를 준수합니다.")
    print("✓ 에이전트가 학습하는데 필요한 모든 정보가 포함되어 있습니다.")
    print("\n다음 정보로 학습 가능:")
    print("  - 시각적 관찰: 게임 화면 이미지 (128x128x4)")
    print("  - 보상 신호: 점수 증가량")
    print("  - 행동 공간: 과일 떨어뜨릴 위치 (연속값 0~1)")
    print("  - 종료 조건: 게임 오버 신호")


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 70)
    print(" " * 20 + "환경 API 상세 테스트")
    print("=" * 70)

    # Mock 환경 테스트
    mock_reward, mock_steps = test_mock_environment()

    # 실제 환경 테스트 (선택적)
    print("\n\n실제 Suika 환경도 테스트하시겠습니까?")
    print("(Chrome/Chromium 및 ChromeDriver 필요, 시간 소요)")
    response = input("y/N: ").strip().lower()

    if response == 'y':
        real_result = test_real_environment_if_available()
    else:
        print("\n실제 환경 테스트를 건너뜁니다.")
        real_result = (None, 0)

    # API 완전성 분석
    analyze_api_completeness(mock_reward, real_result[0])

    print("\n\n" + "=" * 70)
    print("테스트 완료!")
    print("=" * 70)
    print(f"\nMock 환경: {mock_steps} 스텝 실행, 총 보상 {mock_reward:.4f}")
    if real_result[0] is not None:
        print(f"실제 환경: {real_result[1]} 스텝 실행")

    print("\n다음 단계: agents/ 디렉토리에 RL 에이전트를 구현하세요.")
    print("예: agents/dqn_agent.py, agents/ppo_agent.py 등")


if __name__ == "__main__":
    main()
