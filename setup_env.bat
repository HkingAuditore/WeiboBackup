@echo off
REM =====================================================
REM Digital Life Training - Complete Environment Setup
REM For Windows Users
REM =====================================================

echo.
echo ======================================================
echo    Digital Life Training Environment Setup
echo ======================================================
echo.

REM Check Python version
echo [Step 1/5] Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo.
    echo ERROR: Python is not installed!
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation!
    pause
    exit /b 1
)

REM Check pip
echo.
echo [Step 2/5] Checking pip...
pip --version 2>nul
if errorlevel 1 (
    echo ERROR: pip not found!
    pause
    exit /b 1
)

REM Check NVIDIA GPU
echo.
echo [Step 3/5] Checking GPU...
nvidia-smi 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: NVIDIA GPU not detected or driver not installed!
    echo Training on CPU will be VERY slow.
    echo.
    echo Recommended: Install NVIDIA GPU driver from:
    echo   https://www.nvidia.com/drivers
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
)

REM Create virtual environment
echo.
echo [Step 4/5] Creating Python virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created: venv/
) else (
    echo Virtual environment already exists.
)

REM Activate and install packages
echo.
echo [Step 5/5] Installing dependencies...
echo This may take 10-20 minutes...
echo.

call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip -q

REM Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

REM Install other dependencies
echo Installing transformers and training libraries...
pip install transformers>=4.40.0 -q
pip install datasets>=2.18.0 -q
pip install accelerate>=0.28.0 -q
pip install peft>=0.10.0 -q
pip install bitsandbytes>=0.43.0 -q
pip install tensorboard>=2.16.0 -q
pip install scipy>=1.12.0 -q
pip install tqdm rich -q

REM Verify installation
echo.
echo ======================================================
echo Verifying installation...
echo ======================================================
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo.
echo ======================================================
echo    Environment Setup Complete!
echo ======================================================
echo.
echo Next steps:
echo   1. Activate environment: venv\Scripts\activate
echo   2. Prepare training data: python prepare_training_data.py
echo   3. Start training: python finetune_lora.py
echo.
echo Note: The Qwen model will be downloaded automatically
echo       when you run the training script (~14GB for 7B model)
echo.
pause
