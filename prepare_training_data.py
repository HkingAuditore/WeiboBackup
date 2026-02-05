#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Data Preparation Script
Convert weibo texts to conversation format for LLM fine-tuning
"""

import json
import random
import re
from pathlib import Path
from collections import defaultdict


# Define conversation prompts for different scenarios
PROMPTS = {
    'mood': [
        '你现在心情怎么样？',
        '今天感觉如何？',
        '说说你的心情',
        '最近怎么样？',
        '有什么想说的吗？',
    ],
    'daily': [
        '今天做了什么？',
        '在忙什么呢？',
        '最近在干嘛？',
        '有什么新鲜事吗？',
        '说说你的日常',
    ],
    'thought': [
        '你在想什么？',
        '有什么想法？',
        '说说你的看法',
        '你怎么看？',
        '分享一下你的想法',
    ],
    'random': [
        '说点什么吧',
        '随便聊聊',
        '想到什么说什么',
        '来一条微博',
        '发一条状态',
    ]
}


def classify_weibo(text):
    """Classify weibo type based on content"""
    # Mood indicators
    mood_patterns = [
        r'[😭😢😔😞🥺💔难受难过伤心哭泣郁闷烦躁崩溃]',
        r'[😊😄🥰❤️开心快乐幸福美好期待兴奋]',
        r'[累困疲惫无聊空虚迷茫焦虑]',
    ]
    
    # Daily life indicators  
    daily_patterns = [
        r'(今天|昨天|刚才|正在|在.*中)',
        r'(吃|喝|睡|玩|看|听|买|做|写|画|学)',
        r'(出门|回家|上班|下班|出院|入院)',
    ]
    
    # Thought/opinion indicators
    thought_patterns = [
        r'(感觉|觉得|认为|发现|原来|其实|为什么|怎么)',
        r'(希望|想要|想|要是|如果|假如)',
        r'[？?]$',
    ]
    
    for pattern in mood_patterns:
        if re.search(pattern, text):
            return 'mood'
    
    for pattern in daily_patterns:
        if re.search(pattern, text):
            return 'daily'
            
    for pattern in thought_patterns:
        if re.search(pattern, text):
            return 'thought'
    
    return 'random'


def create_alpaca_format(weibos):
    """
    Create Alpaca format training data
    Format: {"instruction": "...", "input": "", "output": "..."}
    """
    data = []
    for weibo in weibos:
        text = weibo['text']
        category = classify_weibo(text)
        prompt = random.choice(PROMPTS[category])
        
        data.append({
            'instruction': prompt,
            'input': '',
            'output': text
        })
    return data


def create_sharegpt_format(weibos):
    """
    Create ShareGPT format training data (multi-turn conversations)
    Format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    """
    data = []
    for weibo in weibos:
        text = weibo['text']
        category = classify_weibo(text)
        prompt = random.choice(PROMPTS[category])
        
        data.append({
            'conversations': [
                {'from': 'human', 'value': prompt},
                {'from': 'gpt', 'value': text}
            ]
        })
    return data


def create_chatml_format(weibos):
    """
    Create ChatML format training data
    Format: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
    """
    system_prompt = """你是一个真实的人，有自己的性格、情绪和表达方式。你的表达风格特点：
- 经常使用emoji和颜文字（如 😊、(´-ω-`)、👴）
- 说话简洁直接，有时带点自嘲
- 喜欢用网络流行语和梗
- 情绪表达真实，不掩饰
- 有时会发牢骚，有时很幽默"""

    data = []
    for weibo in weibos:
        text = weibo['text']
        category = classify_weibo(text)
        prompt = random.choice(PROMPTS[category])
        
        data.append({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': text}
            ]
        })
    return data


def create_multi_turn_conversations(weibos):
    """
    Create multi-turn conversation data by grouping weibos by date
    """
    # Group by date
    by_date = defaultdict(list)
    for weibo in weibos:
        by_date[weibo['date']].append(weibo['text'])
    
    data = []
    for date, texts in by_date.items():
        if len(texts) >= 2:
            # Create a multi-turn conversation
            conversation = []
            for i, text in enumerate(texts):
                if i == 0:
                    prompt = f"今天（{date}）怎么样？"
                else:
                    prompts = ['然后呢？', '还有呢？', '接着呢？', '后来呢？', '继续说说']
                    prompt = random.choice(prompts)
                
                conversation.append({'from': 'human', 'value': prompt})
                conversation.append({'from': 'gpt', 'value': text})
            
            data.append({'conversations': conversation})
    
    return data


def analyze_style(weibos):
    """Analyze writing style for system prompt generation"""
    stats = {
        'avg_length': 0,
        'emoji_usage': 0,
        'question_rate': 0,
        'common_patterns': [],
    }
    
    total_len = 0
    emoji_count = 0
    question_count = 0
    
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0👴🐴]')
    kaomoji_pattern = re.compile(r'[（(][^)）]*[)）]|´.*`|\(.*\)')
    
    for weibo in weibos:
        text = weibo['text']
        total_len += len(text)
        
        if emoji_pattern.search(text) or kaomoji_pattern.search(text):
            emoji_count += 1
        
        if text.endswith('？') or text.endswith('?'):
            question_count += 1
    
    n = len(weibos)
    stats['avg_length'] = total_len / n
    stats['emoji_usage'] = emoji_count / n * 100
    stats['question_rate'] = question_count / n * 100
    
    return stats


def save_jsonl(data, filepath):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    base_dir = Path(__file__).parent
    input_file = base_dir / 'weibo_dataset.jsonl'
    
    print("="*60)
    print("Training Data Preparation Tool")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    weibos = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            weibos.append(json.loads(line.strip()))
    
    print(f"Loaded {len(weibos)} weibos")
    
    # Analyze style
    print("\n" + "-"*40)
    print("Style Analysis")
    print("-"*40)
    style = analyze_style(weibos)
    print(f"  Average length: {style['avg_length']:.1f} chars")
    print(f"  Emoji usage: {style['emoji_usage']:.1f}%")
    print(f"  Question rate: {style['question_rate']:.1f}%")
    
    # Classify weibos
    categories = defaultdict(int)
    for weibo in weibos:
        cat = classify_weibo(weibo['text'])
        categories[cat] += 1
    
    print("\n" + "-"*40)
    print("Content Classification")
    print("-"*40)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(weibos) * 100
        bar = '█' * int(pct / 2)
        print(f"  {cat:8s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Shuffle data
    random.seed(42)
    random.shuffle(weibos)
    
    # Split train/val
    split_idx = int(len(weibos) * 0.9)
    train_weibos = weibos[:split_idx]
    val_weibos = weibos[split_idx:]
    
    print("\n" + "-"*40)
    print("Dataset Split")
    print("-"*40)
    print(f"  Training set: {len(train_weibos)} samples")
    print(f"  Validation set: {len(val_weibos)} samples")
    
    # Generate different formats
    print("\n" + "-"*40)
    print("Generating Training Data Files")
    print("-"*40)
    
    # 1. Alpaca format
    alpaca_train = create_alpaca_format(train_weibos)
    alpaca_val = create_alpaca_format(val_weibos)
    save_jsonl(alpaca_train, base_dir / 'train_alpaca.jsonl')
    save_jsonl(alpaca_val, base_dir / 'val_alpaca.jsonl')
    print(f"  ✓ Alpaca format: train_alpaca.jsonl, val_alpaca.jsonl")
    
    # 2. ShareGPT format
    sharegpt_train = create_sharegpt_format(train_weibos)
    sharegpt_val = create_sharegpt_format(val_weibos)
    save_jsonl(sharegpt_train, base_dir / 'train_sharegpt.jsonl')
    save_jsonl(sharegpt_val, base_dir / 'val_sharegpt.jsonl')
    print(f"  ✓ ShareGPT format: train_sharegpt.jsonl, val_sharegpt.jsonl")
    
    # 3. ChatML format (with system prompt)
    chatml_train = create_chatml_format(train_weibos)
    chatml_val = create_chatml_format(val_weibos)
    save_jsonl(chatml_train, base_dir / 'train_chatml.jsonl')
    save_jsonl(chatml_val, base_dir / 'val_chatml.jsonl')
    print(f"  ✓ ChatML format: train_chatml.jsonl, val_chatml.jsonl")
    
    # 4. Multi-turn conversations
    multi_turn = create_multi_turn_conversations(weibos)
    save_jsonl(multi_turn, base_dir / 'train_multiturn.jsonl')
    print(f"  ✓ Multi-turn format: train_multiturn.jsonl ({len(multi_turn)} conversations)")
    
    # Preview samples
    print("\n" + "="*60)
    print("Sample Preview (ChatML format)")
    print("="*60)
    for i, sample in enumerate(chatml_train[:3], 1):
        print(f"\n--- Sample {i} ---")
        print(f"User: {sample['messages'][1]['content']}")
        print(f"Assistant: {sample['messages'][2]['content']}")
    
    print("\n" + "="*60)
    print("DONE! Training data files generated:")
    print("="*60)
    print("""
Files created:
  📁 train_alpaca.jsonl    - For Alpaca-style fine-tuning
  📁 val_alpaca.jsonl      - Validation set (Alpaca)
  📁 train_sharegpt.jsonl  - For ShareGPT-style training
  📁 val_sharegpt.jsonl    - Validation set (ShareGPT)
  📁 train_chatml.jsonl    - For ChatML format (Qwen, etc.)
  📁 val_chatml.jsonl      - Validation set (ChatML)
  📁 train_multiturn.jsonl - Multi-turn conversations

Recommended next steps:
  1. Use train_chatml.jsonl with Qwen2.5 for best results
  2. Use train_alpaca.jsonl with LLaMA-Factory
  3. Use train_sharegpt.jsonl with general fine-tuning tools
""")


if __name__ == '__main__':
    main()
