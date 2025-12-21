# 프로젝트 아키텍처 문서

## 개요

이 프로젝트는 수박게임(Suika Game)을 플레이하는 강화학습 에이전트를 학습시키기 위한 프레임워크입니다.

## RL 아키텍처 Best Practices

이 프로젝트는 다음 best practices를 따릅니다:

### 1. **환경 추상화** (Gymnasium 표준)
- `envs/suika_wrapper.py`는 Gymnasium의 Wrapper 패턴을 사용
- 실제 환경 구현(suika_rl)을 캡슐화하여 에이전트가 세부사항에 의존하지 않도록 함
- 표준 인터페이스: `reset()`, `step()`, `close()`

### 2. **에이전트 인터페이스** (Strategy 패턴)
- `agents/base_agent.py`에서 추상 베이스 클래스 정의
- 모든 에이전트는 동일한 인터페이스 구현: `select_action()`, `update()`, `save()`, `load()`
- 알고리즘 변경 시 환경이나 학습 루프 수정 불필요

### 3. **설정 기반 접근** (Configuration as Code)
- YAML 설정 파일로 모든 하이퍼파라미터 관리
- 코드 수정 없이 실험 재현 가능
- 버전 관리 가능한 실험 설정

### 4. **모듈화된 학습 루프**
- `training/trainer.py`에서 학습 프로세스 관리
- 에이전트와 환경에 독립적인 학습 로직
- 평가, 체크포인트, 로깅을 통합 관리

### 5. **로깅 및 모니터링**
- TensorBoard, WandB 통합 지원
- 학습 진행 상황 실시간 추적
- 재현 가능한 실험을 위한 메트릭 저장

## 실제 수박게임 구현

### Suika Game의 통합 방식

이 프로젝트는 **외부 오픈소스 수박게임 구현을 API로 연결**하는 방식을 사용합니다:

#### 1. **게임 구현**: TomboFry/suika-game
- JavaScript로 작성된 브라우저 기반 수박게임
- Matter.js 물리 엔진 사용
- 실제 게임과 유사한 물리 시뮬레이션
- 위치: `suika_rl/suika_env/suika-game/`

#### 2. **RL 환경 래퍼**: edwhu/suika_rl
- Selenium WebDriver로 브라우저 게임 제어
- Gymnasium 표준 인터페이스 제공
- 구현 방식:
  1. Python HTTP 서버로 게임 호스팅 (기본 포트: 8923)
  2. Selenium Chrome으로 게임 페이지 열기
  3. 게임 캔버스 스크린샷으로 관찰 수집
  4. JavaScript 인터페이스로 행동 전달

```python
# suika_rl/suika_env/suika_browser_env.py (핵심 부분)
class SuikaBrowserEnv(gymnasium.Env):
    def __init__(self, headless=True, port=8923):
        # HTTP 서버 시작 (게임 호스팅)
        self.server = subprocess.Popen(
            ["python", "-m", "http.server", str(port)],
            cwd=suika_game_dir
        )
        # Selenium Chrome 드라이버 초기화
        self.driver = webdriver.Chrome(options=opts)

    def reset(self):
        # 게임 페이지 로드 및 시작
        self.driver.get(self.game_url)
        self.driver.find_element(By.ID, 'start-game-button').click()

    def step(self, action):
        # 행동을 게임에 전달 (과일 떨어뜨릴 위치)
        self.driver.find_element(By.ID, 'fruit-position').send_keys(action)
        self.driver.find_element(By.ID, 'drop-fruit-button').click()

        # 게임 상태 읽기
        img = canvas.screenshot_as_png  # 관찰 (이미지)
        status, score = self.driver.execute_script(
            'return [window.Game.stateIndex, window.Game.score];'
        )
```

#### 3. **우리 프로젝트의 래퍼**: envs/suika_wrapper.py
- `SuikaBrowserEnv`를 한번 더 감싸서:
  - 관찰 전처리 (정규화, 타입 변환)
  - 보상 함수 커스터마이징
  - 에피소드 통계 추적
  - Mock 환경 fallback 제공

### 데이터 플로우

```
실제 게임 (JavaScript)
        ↕ (DOM 조작, 스크린샷)
Selenium WebDriver
        ↕ (Gymnasium 인터페이스)
SuikaBrowserEnv
        ↕ (전처리, 보상 함수)
SuikaEnvWrapper
        ↕ (학습 알고리즘)
RL Agent (DQN, PPO, etc.)
```

## 설계 결정 및 트레이드오프

### 장점:
1. **실제 게임 물리**: Matter.js로 현실적인 물리 시뮬레이션
2. **디버깅 용이**: 브라우저로 게임 상태 직접 확인 가능
3. **유지보수성**: 게임 로직 수정 시 JavaScript만 변경
4. **분리된 관심사**: 게임 구현 ↔ RL 환경 ↔ 에이전트

### 단점:
1. **속도**: 0.5초/스텝 (Selenium + 렌더링 오버헤드)
2. **의존성**: Chrome/Chromium + ChromeDriver 필요
3. **복잡성**: HTTP 서버 + 브라우저 + Python 3계층

### 개선 방향:
- 게임 물리를 Python으로 재구현 (속도 ↑, 복잡도 ↑)
- 렌더링 비활성화 및 물리 가속 (현재 구조 유지하면서 최적화)
- Headless 브라우저 최적화

## 비교: 다른 접근 방식

### 1. Pure Python 게임 구현 (예: IliasElabbassi/Suika_Watermelon_game)
```
Pygame + Pymunk
```
**장점**: 빠름, 의존성 적음
**단점**: 게임 로직 직접 구현 필요, 원본과 차이 가능

### 2. 우리의 접근 (Selenium + 브라우저 게임)
```
JavaScript 게임 + Selenium + Gymnasium
```
**장점**: 실제 게임과 동일, 디버깅 용이
**단점**: 느림, 의존성 많음

### 3. Stable Baselines3 직접 사용
```
env = make_vec_env('SuikaEnv-v0')
model = PPO('CnnPolicy', env)
model.learn(total_timesteps=1000000)
```
**우리의 접근**: 커스텀 래퍼와 학습 루프를 제공하여 더 세밀한 제어 가능

## RL 프레임워크 비교

### Stable Baselines3 (권장)
- **사용 시점**: 표준 알고리즘(DQN, PPO, SAC) 빠른 프로토타이핑
- **장점**: 검증된 구현, 문서화 우수, 사용 쉬움
- **단점**: 커스터마이징 제한적

### PyTorch 직접 구현 (우리 프로젝트의 베이스)
- **사용 시점**: 새로운 알고리즘 연구, 세밀한 제어 필요
- **장점**: 완전한 제어, 커스터마이징 자유
- **단점**: 버그 가능성, 구현 시간 오래 걸림

### 권장 접근:
1. **프로토타입**: Stable Baselines3로 빠른 검증
2. **연구/개선**: 우리 프레임워크로 커스텀 구현
3. **배포**: 최적화된 커스텀 모델 사용

## 프로젝트 구성 요소 상세

### 환경 (envs/)
```python
SuikaEnvWrapper  # 우리의 래퍼
    └── SuikaBrowserEnv  # suika_rl
        └── 브라우저 게임 (JavaScript)
```

### 에이전트 (agents/)
```
BaseAgent (추상)
    ├── RLAgent (PyTorch 기반)
    │   └── [Your Custom Agent]  # 여기에 구현
    └── RandomAgent (베이스라인)
```

### 학습 (training/)
```
Trainer
    ├── train() - 메인 학습 루프
    ├── evaluate() - 평가
    └── save_checkpoint() - 체크포인트 관리
```

### 설정 (config/)
```yaml
env:      # 환경 설정
agent:    # 에이전트/알고리즘 파라미터
training: # 학습 설정
logging:  # 로깅 설정
```

## 확장 가능성

### 새로운 에이전트 추가:
```python
# agents/my_dqn.py
class DQNAgent(RLAgent):
    def __init__(self, ...):
        # 신경망, 리플레이 버퍼 초기화

    def _forward_policy(self, obs, deterministic):
        # Q-network forward

    def update(self, obs, action, reward, next_obs, done):
        # DQN 학습 알고리즘
```

### 새로운 환경 래퍼:
```python
# envs/custom_wrapper.py
class CustomRewardWrapper(SuikaEnvWrapper):
    def _process_reward(self, reward, info):
        # 커스텀 보상 함수
        return custom_reward
```

### 병렬 환경:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

envs = [make_suika_env for _ in range(4)]
vec_env = SubprocVecEnv(envs)
```

## 참고 자료

### 사용된 오픈소스
- [TomboFry/suika-game](https://github.com/TomboFry/suika-game) - 게임 구현
- [edwhu/suika_rl](https://github.com/edwhu/suika_rl) - RL 환경
- [Gymnasium](https://gymnasium.farama.org/) - RL 표준 인터페이스
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - RL 알고리즘

### Best Practices 참고
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Environment Creation](https://gymnasium.farama.org/introduction/create_custom_env/)
