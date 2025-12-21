@echo off
REM API 테스트 실행 스크립트 (Windows)

echo =========================================
echo Suika RL API 테스트 스크립트
echo =========================================

REM 가상환경 확인
if exist venv\ (
    echo √ 가상환경 발견
    call venv\Scripts\activate.bat
) else (
    echo 가상환경이 없습니다. 생성 중...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM 최소 의존성 설치
echo.
echo 필수 패키지 설치 중...
pip install -q numpy gymnasium pillow

echo.
echo =========================================
echo 간단한 테스트 실행
echo =========================================
python tests\test_simple.py

echo.
echo =========================================
echo 상세한 API 테스트 실행
echo =========================================
python tests\test_environment_api.py

pause
