#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weibo Data Extraction Script
Extract all weibo texts from HTML backup files for digital life training
"""

import os
import re
import json
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("Warning: BeautifulSoup not found, using regex fallback")


def extract_with_bs4(html_content):
    """Extract weibo texts using BeautifulSoup"""
    soup = BeautifulSoup(html_content, 'html.parser')
    texts = []
    for weibo_text_div in soup.find_all('div', class_='weibo-text'):
        text = weibo_text_div.get_text(strip=True)
        if text:
            texts.append(text)
    return texts


def extract_with_regex(html_content):
    """Fallback: Extract weibo texts using regex"""
    # Match content between <div class="weibo-text"> and </div>
    pattern = r'<div class="weibo-text"[^>]*>(.*?)</div>'
    matches = re.findall(pattern, html_content, re.DOTALL)
    texts = []
    for match in matches:
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', match)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            texts.append(text)
    return texts


def extract_weibo_texts(base_dir):
    """Extract all weibo texts from HTML files"""
    all_weibos = []
    stats = {
        'total_files': 0,
        'total_weibos': 0,
        'by_year': {},
        'filtered_out': 0
    }
    
    # Filter list - content to skip
    filter_list = ['分享图片', '转发微博', '轉發微博', '']
    
    base_path = Path(base_dir)
    
    # Find all year directories
    for year_dir in sorted(base_path.iterdir()):
        if year_dir.is_dir() and year_dir.name.isdigit():
            html_dir = year_dir / "html"
            if html_dir.exists():
                year_count = 0
                
                for html_file in sorted(html_dir.glob("*.html")):
                    # Skip index files
                    if html_file.name == "index.html":
                        continue
                    
                    stats['total_files'] += 1
                    
                    try:
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                    except Exception as e:
                        print(f"Error reading {html_file}: {e}")
                        continue
                    
                    # Extract date from filename
                    date = html_file.stem  # e.g., "2020-03-05"
                    
                    # Extract texts
                    if HAS_BS4:
                        texts = extract_with_bs4(html_content)
                    else:
                        texts = extract_with_regex(html_content)
                    
                    for text in texts:
                        # Filter out meaningless content
                        if text in filter_list:
                            stats['filtered_out'] += 1
                            continue
                        
                        # Skip very short texts (likely noise)
                        if len(text) < 2:
                            stats['filtered_out'] += 1
                            continue
                        
                        all_weibos.append({
                            'date': date,
                            'year': year_dir.name,
                            'text': text
                        })
                        year_count += 1
                
                stats['by_year'][year_dir.name] = year_count
                stats['total_weibos'] += year_count
    
    return all_weibos, stats


def clean_text(text):
    """Clean and normalize text"""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def save_to_jsonl(weibos, output_file):
    """Save to JSONL format (suitable for training)"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for weibo in weibos:
            weibo['text'] = clean_text(weibo['text'])
            if weibo['text']:
                f.write(json.dumps(weibo, ensure_ascii=False) + '\n')


def print_sample_weibos(weibos, n=10):
    """Print sample weibos for preview"""
    print(f"\n{'='*60}")
    print(f"Sample Weibos (showing {n} random samples):")
    print('='*60)
    
    import random
    samples = random.sample(weibos, min(n, len(weibos)))
    
    for i, weibo in enumerate(samples, 1):
        text_preview = weibo['text'][:80] + '...' if len(weibo['text']) > 80 else weibo['text']
        print(f"\n[{i}] ({weibo['date']})")
        print(f"    {text_preview}")


def main():
    base_dir = Path(__file__).parent
    
    print("="*60)
    print("Weibo Data Extraction Tool")
    print("="*60)
    print(f"\nScanning directory: {base_dir}")
    
    # Extract data
    print("\nExtracting weibo data from HTML files...")
    weibos, stats = extract_weibo_texts(base_dir)
    
    # Print statistics
    print("\n" + "="*60)
    print("EXTRACTION STATISTICS")
    print("="*60)
    print(f"\nTotal HTML files processed: {stats['total_files']}")
    print(f"Total valid weibos extracted: {stats['total_weibos']}")
    print(f"Filtered out (noise/duplicates): {stats['filtered_out']}")
    
    print("\nBreakdown by year:")
    print("-"*30)
    for year, count in sorted(stats['by_year'].items()):
        bar = '█' * (count // 50)  # Simple bar chart
        print(f"  {year}: {count:>5} weibos {bar}")
    
    # Show samples
    if weibos:
        print_sample_weibos(weibos)
    
    # Save to file
    output_file = base_dir / 'weibo_dataset.jsonl'
    save_to_jsonl(weibos, output_file)
    print(f"\n{'='*60}")
    print(f"Data saved to: {output_file}")
    print(f"{'='*60}")
    
    return weibos, stats


if __name__ == '__main__':
    main()
