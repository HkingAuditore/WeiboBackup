@echo off
REM Quick start script for Digital Life training (Windows)

echo ==============================================
echo  Digital Life LoRA Fine-tuning
echo ==============================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed!
    exit /b 1
)

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt -q

REM Check training data
if not exist "train_chatml.jsonl" (
    echo.
    echo Training data not found. Generating...
    python prepare_training_data.py
)

REM Start training
echo.
echo Starting training with default settings...
echo   Model: Qwen/Qwen2.5-7B-Instruct
echo   Data: train_chatml.jsonl
echo   Method: QLoRA (4-bit)
echo.

python finetune_lora.py ^
    --model_name "Qwen/Qwen2.5-7B-Instruct" ^
    --train_data "train_chatml.jsonl" ^
    --val_data "val_chatml.jsonl" ^
    --output_dir "./output_lora" ^
    --epochs 3 ^
    --batch_size 4 ^
    --gradient_accumulation 4 ^
    --lora_r 64 ^
    --lora_alpha 128 ^
    --use_4bit

echo.
echo Training complete! Check ./output_lora for results.
pause
