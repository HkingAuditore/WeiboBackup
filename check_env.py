#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Environment Checker - Verify training environment is ready
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def check_python():
    """Check Python version"""
    print_header("Python Environment")
    
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"  Executable: {sys.executable}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("  ⚠️  WARNING: Python 3.9+ is recommended")
        return False
    
    print("  ✓ Python version OK")
    return True


def check_pytorch():
    """Check PyTorch installation"""
    print_header("PyTorch")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({vram_gb:.1f} GB)")
            print("  ✓ CUDA is available!")
        else:
            print("  ⚠️  CUDA not available - training will be slow")
            
        return True
    except ImportError:
        print("  ✗ PyTorch not installed")
        print("  Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False


def check_transformers():
    """Check Transformers installation"""
    print_header("Transformers & Related Packages")
    
    packages = [
        ('transformers', '4.40.0'),
        ('datasets', '2.18.0'),
        ('accelerate', '0.28.0'),
        ('peft', '0.10.0'),
        ('bitsandbytes', '0.43.0'),
    ]
    
    all_ok = True
    for package, min_version in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space"""
    print_header("Disk Space")
    
    import shutil
    
    path = Path(__file__).parent
    total, used, free = shutil.disk_usage(path)
    
    total_gb = total / 1024**3
    free_gb = free / 1024**3
    
    print(f"  Location: {path}")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Free: {free_gb:.1f} GB")
    
    if free_gb < 30:
        print("  ⚠️  WARNING: Less than 30GB free space")
        print("      Recommended: 50GB+ for 7B model training")
        return False
    else:
        print("  ✓ Sufficient disk space")
        return True


def check_training_data():
    """Check if training data exists"""
    print_header("Training Data")
    
    base_dir = Path(__file__).parent
    
    data_files = [
        ('weibo_dataset.jsonl', 'Source data'),
        ('train_chatml.jsonl', 'Training data (ChatML)'),
        ('val_chatml.jsonl', 'Validation data (ChatML)'),
    ]
    
    all_ok = True
    for filename, desc in data_files:
        filepath = base_dir / filename
        if filepath.exists():
            # Count lines
            with open(filepath, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename}: {count} samples ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filename}: NOT FOUND")
            if filename != 'weibo_dataset.jsonl':
                all_ok = False
    
    if not (base_dir / 'train_chatml.jsonl').exists():
        print("\n  Run: python prepare_training_data.py")
    
    return all_ok


def estimate_training_requirements(model_size="7B"):
    """Estimate training requirements"""
    print_header(f"Training Requirements ({model_size} model)")
    
    requirements = {
        "3B": {"vram": 8, "ram": 16, "disk": 20, "time": "1-2 hours"},
        "7B": {"vram": 16, "ram": 32, "disk": 50, "time": "2-4 hours"},
        "14B": {"vram": 24, "ram": 64, "disk": 100, "time": "4-8 hours"},
    }
    
    req = requirements.get(model_size, requirements["7B"])
    
    print(f"  Minimum VRAM: {req['vram']} GB (with 4-bit quantization)")
    print(f"  Recommended RAM: {req['ram']} GB")
    print(f"  Disk space: {req['disk']} GB")
    print(f"  Estimated time: {req['time']} (for 3 epochs)")
    
    # Check if current system meets requirements
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram >= req['vram']:
                print(f"\n  ✓ Your GPU ({vram:.1f} GB) meets the requirements!")
            else:
                print(f"\n  ⚠️  Your GPU ({vram:.1f} GB) may not be sufficient")
                print(f"     Consider using a smaller model (3B) or cloud training")
    except:
        pass


def main():
    print("\n" + "="*60)
    print("  🔍 Digital Life Training - Environment Check")
    print("="*60)
    
    results = []
    
    results.append(("Python", check_python()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Transformers", check_transformers()))
    results.append(("Disk Space", check_disk_space()))
    results.append(("Training Data", check_training_data()))
    
    estimate_training_requirements("7B")
    
    # Summary
    print_header("Summary")
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n  🎉 All checks passed! Ready to train.")
        print("\n  Next steps:")
        print("    1. python prepare_training_data.py  (if not done)")
        print("    2. python finetune_lora.py")
    else:
        print("\n  ⚠️  Some checks failed. Please fix the issues above.")
        print("\n  Quick fix:")
        print("    1. Run setup_env.bat to install dependencies")
        print("    2. Make sure NVIDIA driver is installed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
