#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference Script for Digital Life Model
Chat with your fine-tuned model
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, base_model: str = None):
    """Load fine-tuned model"""
    
    model_path = Path(model_path)
    
    # Check if it's a LoRA adapter or merged model
    adapter_config = model_path / "adapter_config.json"
    is_lora = adapter_config.exists()
    
    if is_lora:
        if not base_model:
            # Try to get base model from config
            import json
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    else:
        print(f"Loading merged model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    
    model.eval()
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """Generate response from model"""
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def chat_loop(model, tokenizer, system_prompt: str = None):
    """Interactive chat loop"""
    
    if system_prompt is None:
        system_prompt = """你是一个真实的人，有自己的性格、情绪和表达方式。你的表达风格特点：
- 经常使用emoji和颜文字（如 😊、(´-ω-`)、👴）
- 说话简洁直接，有时带点自嘲
- 喜欢用网络流行语和梗
- 情绪表达真实，不掩饰
- 有时会发牢骚，有时很幽默"""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    print("\n" + "="*60)
    print("🧬 Digital Life Chat")
    print("="*60)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("Type 'system <prompt>' to change system prompt")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Bye! 👋")
            break
        
        if user_input.lower() == 'clear':
            messages = [{"role": "system", "content": system_prompt}]
            print("[Conversation cleared]\n")
            continue
        
        if user_input.lower().startswith('system '):
            system_prompt = user_input[7:].strip()
            messages = [{"role": "system", "content": system_prompt}]
            print(f"[System prompt updated]\n")
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        print("AI: ", end="", flush=True)
        response = generate_response(model, tokenizer, messages)
        print(response + "\n")
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Chat with Digital Life Model")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model or LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (for LoRA adapter)")
    parser.add_argument("--system_prompt", type=str, default=None,
                       help="Custom system prompt")
    parser.add_argument("--single", type=str, default=None,
                       help="Single query mode (no interactive chat)")
    
    args = parser.parse_args()
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    print("Model loaded!\n")
    
    if args.single:
        # Single query mode
        messages = [
            {"role": "system", "content": args.system_prompt or "你是一个真实的人。"},
            {"role": "user", "content": args.single}
        ]
        response = generate_response(model, tokenizer, messages)
        print(f"Response: {response}")
    else:
        # Interactive chat mode
        chat_loop(model, tokenizer, args.system_prompt)


if __name__ == "__main__":
    main()
