# train_model.py - 训练模型
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
from config import Config
from word2vec_model import SkipGramNS

def check_indices(data, vocab_size, name):
    """检查索引是否在有效范围内"""
    max_idx = np.max(data)
    min_idx = np.min(data)
    print(f"{name} - 最小值: {min_idx}, 最大值: {max_idx}, 词汇表大小: {vocab_size}")
    
    if min_idx < 0:
        print(f"错误: {name} 中有负索引")
        return False
    
    if max_idx >= vocab_size:
        print(f"错误: {name} 中有索引 {max_idx} >= 词汇表大小 {vocab_size}")
        return False
    
    return True

def train_model(targets, contexts, neg_samples, vocab_size):
    """训练模型"""
    # 检查索引范围
    if not check_indices(targets, vocab_size, "目标词索引"):
        return
    if not check_indices(contexts, vocab_size, "上下文词索引"):
        return
    if not check_indices(neg_samples, vocab_size, "负样本索引"):
        return
    
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = SkipGramNS(vocab_size, Config.EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 准备数据加载器
    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(targets),
        torch.LongTensor(contexts),
        torch.LongTensor(neg_samples)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    
    # 训练循环
    losses = []
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        for batch in progress:
            target, context, neg_samples = batch
            target = target.to(device)
            context = context.to(device)
            neg_samples = neg_samples.to(device)
            
            optimizer.zero_grad()
            loss = model(target, context, neg_samples)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, Config.NUM_EPOCHS+1), losses, 'o-')
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('outputs/training_loss.png')
    plt.close()
    print("训练损失图已保存到 outputs/training_loss.png")
    
    # 保存词向量
    embeddings = model.get_embeddings()
    np.save('outputs/word_embeddings.npy', embeddings)
    print("词向量已保存到 outputs/word_embeddings.npy")
    
    return embeddings

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 加载训练数据
    targets = np.load('outputs/targets.npy')
    contexts = np.load('outputs/contexts.npy')
    neg_samples = np.load('outputs/neg_samples.npy')
    
    print(f"加载的训练数据:")
    print(f" - 目标词数量: {len(targets)}")
    print(f" - 上下文词数量: {len(contexts)}")
    print(f" - 负样本集数量: {len(neg_samples)}")
    
    # 加载词汇表以获取词汇表大小
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 训练模型
    train_model(targets, contexts, neg_samples, vocab_size)