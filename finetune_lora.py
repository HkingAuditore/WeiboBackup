#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA Fine-tuning Script for Digital Life Training
Supports Qwen2.5, LLaMA, and other transformer models
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


@dataclass
class ModelConfig:
    """Model and training configuration"""
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 512
    use_4bit: bool = True
    use_8bit: bool = False
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 64                    # LoRA rank
    lora_alpha: int = 128          # LoRA alpha
    lora_dropout: float = 0.05    # Dropout rate
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def load_and_preprocess_data(tokenizer, data_path, max_length=512):
    """Load and preprocess ChatML format data"""
    
    def preprocess_function(examples):
        """Convert ChatML messages to model input format"""
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for messages in examples["messages"]:
            # Build conversation text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            
            # For causal LM, labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            model_inputs["input_ids"].append(tokenized["input_ids"])
            model_inputs["attention_mask"].append(tokenized["attention_mask"])
            model_inputs["labels"].append(tokenized["labels"])
        
        return model_inputs
    
    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    # Preprocess
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing dataset",
    )
    
    return processed_dataset


def create_model_and_tokenizer(config: ModelConfig):
    """Initialize model and tokenizer with quantization"""
    
    print(f"\n{'='*60}")
    print(f"Loading model: {config.model_name}")
    print(f"{'='*60}")
    
    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA)")
    elif config.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
    )
    
    # Prepare for k-bit training
    if config.use_4bit or config.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    model.config.use_cache = False
    
    return model, tokenizer


def apply_lora(model, lora_config: LoRAConfig):
    """Apply LoRA adapter to model"""
    
    print(f"\n{'='*60}")
    print("Applying LoRA Configuration")
    print(f"{'='*60}")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def train(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    logging_steps: int = 10,
    save_steps: int = 100,
):
    """Run training"""
    
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset) if val_dataset else 0}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=save_steps if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")
    
    # Train
    trainer.train()
    
    # Save final model
    print(f"\n{'='*60}")
    print("Saving Model...")
    print(f"{'='*60}")
    
    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"Model saved to: {final_dir}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Digital Life")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="train_chatml.jsonl",
                       help="Training data file (ChatML format)")
    parser.add_argument("--val_data", type=str, default="val_chatml.jsonl",
                       help="Validation data file")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./output_lora",
                       help="Output directory for checkpoints")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=64,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    
    # Quantization
    parser.add_argument("--use_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--no_quant", action="store_true",
                       help="Disable quantization (requires more VRAM)")
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("🧬 Digital Life LoRA Fine-tuning")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent
    train_path = base_dir / args.train_data
    val_path = base_dir / args.val_data if args.val_data else None
    
    # Check files exist
    if not train_path.exists():
        print(f"Error: Training data not found: {train_path}")
        print("Please run prepare_training_data.py first!")
        return
    
    # Setup configs
    model_config = ModelConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        use_4bit=args.use_4bit and not args.no_quant,
        use_8bit=args.use_8bit and not args.no_quant,
    )
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Initialize model and tokenizer
    model, tokenizer = create_model_and_tokenizer(model_config)
    
    # Apply LoRA
    model = apply_lora(model, lora_config)
    
    # Load data
    print(f"\n{'='*60}")
    print("Loading Training Data")
    print(f"{'='*60}")
    
    train_dataset = load_and_preprocess_data(
        tokenizer, train_path, args.max_length
    )
    print(f"Training samples: {len(train_dataset)}")
    
    val_dataset = None
    if val_path and val_path.exists():
        val_dataset = load_and_preprocess_data(
            tokenizer, val_path, args.max_length
        )
        print(f"Validation samples: {len(val_dataset)}")
    
    # Train
    trainer = train(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"""
Next steps:
  1. Test inference: python inference.py --model_path {args.output_dir}/final
  2. Merge LoRA weights: python merge_lora.py --model_path {args.output_dir}/final
  3. Export to other formats (GGUF, etc.)
""")


if __name__ == "__main__":
    main()
