#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge LoRA adapter with base model
Creates a standalone model without requiring PEFT at inference
"""

import argparse
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(
    base_model: str,
    lora_path: str,
    output_path: str,
    push_to_hub: bool = False,
    hub_repo: str = None,
):
    """Merge LoRA adapter with base model"""
    
    print("\n" + "="*60)
    print("🔀 LoRA Merge Tool")
    print("="*60)
    
    lora_path = Path(lora_path)
    output_path = Path(output_path)
    
    # Auto-detect base model from adapter config
    if not base_model:
        adapter_config = lora_path / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
        
        if not base_model:
            raise ValueError("Base model not specified and cannot be detected from adapter config")
    
    print(f"\nBase model: {base_model}")
    print(f"LoRA adapter: {lora_path}")
    print(f"Output path: {output_path}")
    
    # Load base model
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    # Merge and unload
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(
        output_path,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_path)
    
    # Optionally push to Hub
    if push_to_hub and hub_repo:
        print(f"Pushing to Hugging Face Hub: {hub_repo}")
        model.push_to_hub(hub_repo)
        tokenizer.push_to_hub(hub_repo)
    
    print("\n" + "="*60)
    print("✅ Merge Complete!")
    print("="*60)
    print(f"""
Merged model saved to: {output_path}

You can now use the merged model directly:
  python inference.py --model_path {output_path}

Or convert to other formats:
  - GGUF (for llama.cpp): Use llama.cpp's convert.py
  - AWQ: Use AutoAWQ
  - GPTQ: Use AutoGPTQ
""")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (auto-detected if not specified)")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA adapter")
    parser.add_argument("--output_path", type=str, default="./merged_model",
                       help="Output path for merged model")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push merged model to Hugging Face Hub")
    parser.add_argument("--hub_repo", type=str, default=None,
                       help="Hugging Face Hub repository name")
    
    args = parser.parse_args()
    
    merge_lora(
        base_model=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )


if __name__ == "__main__":
    main()
