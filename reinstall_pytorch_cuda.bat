@echo off
REM Reinstall PyTorch with CUDA support
echo Uninstalling current PyTorch (CPU version)...
call venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio

echo.
echo Installing PyTorch with CUDA 12.1 support...
call venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Verifying installation...
call venv\Scripts\python.exe -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA not available'); print(f'Device count: {torch.cuda.device_count()}'); [print(f'Device {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo.
echo Done!
pause
