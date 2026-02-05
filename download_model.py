#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Download Script
Download Qwen2.5 models from Hugging Face or ModelScope
"""

import os
import sys
import argparse
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ Transformers {transformers.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please run: pip install torch transformers")
        return False


def download_from_huggingface(model_name, output_dir):
    """Download model from Hugging Face"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    
    print(f"\n{'='*60}")
    print(f"Downloading from Hugging Face: {model_name}")
    print(f"{'='*60}")
    
    output_path = Path(output_dir) / model_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    print("\nDownloading model files (this may take a while)...")
    
    # Download using snapshot_download for better resume support
    snapshot_download(
        repo_id=model_name,
        local_dir=str(output_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    
    print(f"\n✓ Model downloaded to: {output_path}")
    return output_path


def download_from_modelscope(model_name, output_dir):
    """Download model from ModelScope (for users in China)"""
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError:
        print("ModelScope not installed. Installing...")
        os.system(f"{sys.executable} -m pip install modelscope -q")
        from modelscope import snapshot_download as ms_download
    
    # Map Hugging Face names to ModelScope names
    modelscope_mapping = {
        "Qwen/Qwen2.5-3B-Instruct": "qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct": "qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct": "qwen/Qwen2.5-14B-Instruct",
    }
    
    ms_model_name = modelscope_mapping.get(model_name, model_name)
    
    print(f"\n{'='*60}")
    print(f"Downloading from ModelScope: {ms_model_name}")
    print(f"{'='*60}")
    
    output_path = Path(output_dir) / model_name.replace("/", "_")
    
    model_dir = ms_download(
        ms_model_name,
        cache_dir=str(output_path.parent),
    )
    
    print(f"\n✓ Model downloaded to: {model_dir}")
    return model_dir


def verify_model(model_path):
    """Verify downloaded model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n{'='*60}")
    print("Verifying model...")
    print(f"{'='*60}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
        
        print("Loading model config...")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Model type: {config.model_type}")
        print(f"✓ Hidden size: {config.hidden_size}")
        print(f"✓ Num layers: {config.num_hidden_layers}")
        
        print("\n✓ Model verification passed!")
        return True
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Qwen2.5 models")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       choices=[
                           "Qwen/Qwen2.5-3B-Instruct",
                           "Qwen/Qwen2.5-7B-Instruct", 
                           "Qwen/Qwen2.5-14B-Instruct",
                       ],
                       help="Model to download")
    parser.add_argument("--output_dir", type=str, default="./models",
                       help="Output directory for downloaded model")
    parser.add_argument("--source", type=str, default="auto",
                       choices=["huggingface", "modelscope", "auto"],
                       help="Download source")
    parser.add_argument("--verify", action="store_true",
                       help="Verify model after download")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🤗 Model Download Tool")
    print("="*60)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Determine download source
    source = args.source
    if source == "auto":
        # Try to detect if user is in China
        try:
            import socket
            socket.setdefaulttimeout(3)
            socket.socket().connect(("huggingface.co", 443))
            source = "huggingface"
            print("\n✓ Hugging Face accessible, using HF as source")
        except:
            source = "modelscope"
            print("\n! Hugging Face not accessible, using ModelScope")
    
    # Model size info
    model_sizes = {
        "Qwen/Qwen2.5-3B-Instruct": "~6GB",
        "Qwen/Qwen2.5-7B-Instruct": "~14GB",
        "Qwen/Qwen2.5-14B-Instruct": "~28GB",
    }
    
    print(f"\nModel: {args.model}")
    print(f"Size: {model_sizes.get(args.model, 'Unknown')}")
    print(f"Source: {source}")
    
    # Download
    if source == "huggingface":
        model_path = download_from_huggingface(args.model, args.output_dir)
    else:
        model_path = download_from_modelscope(args.model, args.output_dir)
    
    # Verify
    if args.verify:
        verify_model(model_path)
    
    print("\n" + "="*60)
    print("✅ Download Complete!")
    print("="*60)
    print(f"""
Model location: {model_path}

To use this model for training:
  python finetune_lora.py --model_name "{model_path}"

Or set HF_HOME environment variable:
  set HF_HOME={Path(args.output_dir).absolute()}  (Windows)
  export HF_HOME={Path(args.output_dir).absolute()}  (Linux/Mac)
""")


if __name__ == "__main__":
    main()
