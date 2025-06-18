# data_generator.py - 生成训练数据
import numpy as np
import pickle
from tqdm import tqdm
import config
import os

def generate_training_data(tokens, vocab, neg_weights):
    """生成训练数据"""
    # 将文本转换为索引
    word_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    vocab_size = len(vocab)
    
    # 生成正样本 (target, context) 和负样本
    targets = []
    contexts = []
    neg_samples_list = []
    
    print("生成训练数据...")
    for i, target_id in tqdm(enumerate(word_ids), total=len(word_ids)):
        start = max(0, i - config.Config.WINDOW_SIZE)
        end = min(len(word_ids), i + config.Config.WINDOW_SIZE + 1)
        
        for j in range(start, end):
            if i != j:  # 跳过目标词本身
                context_id = word_ids[j]
                # 负采样 - 确保索引在范围内
                neg_samples = np.random.choice(
                    vocab_size,  # 使用词汇表大小而不是len(neg_weights)
                    size=config.Config.NUM_NEG_SAMPLES, 
                    p=neg_weights,
                    replace=False
                )
                
                targets.append(target_id)
                contexts.append(context_id)
                neg_samples_list.append(neg_samples)
    
    print(f"生成的训练样本数: {len(targets)}")
    return targets, contexts, neg_samples_list

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 加载预处理后的tokens
    with open('data/preprocessed_tokens.txt', 'r', encoding='utf-8') as f:
        tokens = f.read().split()
    
    # 加载词汇表和负采样分布
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    neg_weights = np.load('outputs/neg_weights.npy')
    
    # 生成训练数据
    targets, contexts, neg_samples = generate_training_data(tokens, vocab, neg_weights)
    
    # 保存训练数据
    np.save('outputs/targets.npy', np.array(targets))
    np.save('outputs/contexts.npy', np.array(contexts))
    np.save('outputs/neg_samples.npy', np.array(neg_samples))
    
    print("训练数据已保存到 outputs/ 目录")
    print(f" - targets.npy: {len(targets)} 个样本")
    print(f" - contexts.npy: {len(contexts)} 个样本")
    print(f" - neg_samples.npy: {len(neg_samples)} 个样本")