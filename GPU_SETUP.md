# GPU 설정 가이드

## 문제 상황

현재 두 가지 문제가 발생하고 있습니다:

1. **GPU 인식 실패**: PyTorch가 CPU 버전으로 설치되어 있어 GPU를 사용하지 못함
2. **이미지 크기 불일치 (해결됨)**: 서버 코드는 수정되었으나, 실행 중인 서버를 재시작해야 함

## 해결 방법

### 1. PyTorch CUDA 버전 재설치

현재 PyTorch CPU 버전(`2.9.1+cpu`)이 설치되어 있습니다. NVIDIA GPU를 사용하려면 CUDA 버전을 설치해야 합니다.

#### 방법 A: 배치 파일 사용 (권장)

프로젝트 루트에서 `reinstall_pytorch_cuda.bat`을 더블클릭하거나 CMD에서 실행:

```cmd
reinstall_pytorch_cuda.bat
```

#### 방법 B: 수동 설치

Windows PowerShell이나 CMD에서 다음 명령을 순서대로 실행:

```cmd
cd D:\hj\suika-agent

# 1. 기존 PyTorch 제거
venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

# 2. CUDA 12.1 버전 설치
venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 설치 확인
venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**CUDA 버전 확인:**
- CUDA 11.8을 사용 중이라면: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1을 사용 중이라면: `https://download.pytorch.org/whl/cu121`

NVIDIA 드라이버에 포함된 CUDA 버전 확인:
```cmd
nvidia-smi
```

### 2. 게임 서버 재시작

이미지 크기 문제를 해결하기 위해 Node.js 서버를 재시작해야 합니다.

#### 서버 종료
현재 실행 중인 서버를 종료:
- Ctrl+C로 종료하거나
- 작업 관리자에서 `node.exe` 프로세스 종료

#### 서버 시작
```cmd
cd suika_rl\server
npm start
```

또는 개발 모드 (자동 재시작):
```cmd
npm run dev
```

### 3. 학습 재실행

서버가 재시작되고 PyTorch CUDA가 설치된 후:

```cmd
venv\Scripts\python.exe main.py --config config/default.yaml
```

정상 작동 시 다음과 같이 출력됩니다:
```
Device: GPU - NVIDIA GeForce RTX XXXX (XX.X GB)
```

## 확인 사항

### GPU 인식 확인
```cmd
venv\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

예상 출력:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

### 서버 이미지 크기 확인
```cmd
cd suika_rl\server
node -e "const {SuikaGame} = require('./game'); const g = new SuikaGame(); console.log('Image size:', g.getObservation().image.length, '(expected: 399360)');"
```

예상 출력:
```
Image size: 399360 (expected: 399360)
```

## 트러블슈팅

### PyTorch 설치 실패
- 인터넷 연결 확인
- pip 업데이트: `python -m pip install --upgrade pip`
- 디스크 공간 확인 (CUDA 버전은 ~2GB)

### GPU 여전히 인식 안됨
1. NVIDIA 드라이버 업데이트
2. CUDA Toolkit 설치 확인
3. 시스템 재부팅

### 서버 연결 실패
1. 포트 8924가 사용 중인지 확인: `netstat -ano | findstr :8924`
2. 방화벽 설정 확인
3. Node.js 버전 확인: `node --version` (v16 이상 권장)

## 추가 정보

- PyTorch 공식 설치 가이드: https://pytorch.org/get-started/locally/
- CUDA 호환성: https://pytorch.org/get-started/locally/#windows-prerequisites
