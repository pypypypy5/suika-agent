#!/bin/bash
# API 테스트 실행 스크립트

echo "========================================="
echo "Suika RL API 테스트 스크립트"
echo "========================================="

# 가상환경 확인
if [ -d "venv" ]; then
    echo "✓ 가상환경 발견"
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
else
    echo "가상환경이 없습니다. 생성 중..."
    python3 -m venv venv || python -m venv venv
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
fi

# 최소 의존성 설치
echo ""
echo "필수 패키지 설치 중..."
pip install -q numpy gymnasium pillow

echo ""
echo "========================================="
echo "간단한 테스트 실행"
echo "========================================="
python tests/test_simple.py

echo ""
echo "========================================="
echo "상세한 API 테스트 실행"
echo "========================================="
python tests/test_environment_api.py
