# vocab_builder.py - 构建词汇表和负采样分布
from collections import Counter
import numpy as np
import pickle
import os
from config import Config

def build_vocab(tokens):
    """构建词汇表"""
    counter = Counter(tokens)
    
    # 过滤低频词
    vocab = {}
    idx = 0
    for word, count in counter.items():
        if count >= Config.MIN_COUNT:
            vocab[word] = idx
            idx += 1
    
    # 添加未知词标记
    vocab['<unk>'] = idx
    
    # 创建反向映射
    idx_to_word = {index: word for word, index in vocab.items()}
    
    print(f"词汇表大小: {len(vocab)}")
    print(f"前10个高频词: {counter.most_common(10)}")
    
    # 验证索引
    max_index = max(vocab.values())
    if max_index != len(vocab) - 1:
        print(f"警告: 最大索引({max_index})不等于词汇表大小减1({len(vocab)-1})")
    
    return vocab, idx_to_word

def create_negative_sampling_distribution(tokens, vocab):
    """创建负采样分布"""
    # 只考虑在词汇表中的词
    word_ids = [vocab[word] for word in tokens if word in vocab]
    id_counts = Counter(word_ids)
    
    # 调试信息：打印词汇表大小和最大索引
    vocab_size = len(vocab)
    max_id = max(id_counts.keys()) if id_counts else 0
    print(f"词汇表大小: {vocab_size}, 最大索引: {max_id}")
    
    # 计算每个词的权重
    weights = np.zeros(vocab_size)
    valid_ids = [idx for idx in id_counts.keys() if idx < vocab_size]
    
    if not valid_ids:
        print("警告：没有有效的词ID")
        return weights
    
    for idx in valid_ids:
        count = id_counts[idx]
        weights[idx] = count ** 0.75
    
    # 归一化
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        print("警告：权重总和为0")
        # 使用均匀分布作为后备
        weights = np.ones(vocab_size) / vocab_size
    
    return weights

if __name__ == "__main__":
    os.makedirs('outputs', exist_ok=True)
    
    # 加载预处理后的tokens
    with open('data/preprocessed_tokens.txt', 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    
    print(f"总tokens数: {len(tokens)}")
    
    # 构建词汇表
    vocab, idx_to_word = build_vocab(tokens)
    
    # 创建负采样分布
    neg_weights = create_negative_sampling_distribution(tokens, vocab)
    
    # 保存结果
    with open('outputs/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('outputs/idx_to_word.pkl', 'wb') as f:
        pickle.dump(idx_to_word, f)
    np.save('outputs/neg_weights.npy', neg_weights)
    
    print("词汇表和负采样分布已保存到 outputs/ 目录")