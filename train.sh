#!/bin/bash
# Quick start script for Digital Life training

echo "=============================================="
echo "🧬 Digital Life LoRA Fine-tuning"
echo "=============================================="

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed!"
    exit 1
fi

# Check CUDA
if ! python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "Warning: CUDA is not available. Training will be slow on CPU."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Check training data
if [ ! -f "train_chatml.jsonl" ]; then
    echo ""
    echo "Training data not found. Generating..."
    python prepare_training_data.py
fi

# Start training
echo ""
echo "Starting training with default settings..."
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  Data: train_chatml.jsonl"
echo "  Method: QLoRA (4-bit)"
echo ""

python finetune_lora.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "train_chatml.jsonl" \
    --val_data "val_chatml.jsonl" \
    --output_dir "./output_lora" \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation 4 \
    --lora_r 64 \
    --lora_alpha 128 \
    --use_4bit

echo ""
echo "Training complete! Check ./output_lora for results."
