# train_model.py - 训练模型
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from config import Config
from word2vec_model import SkipGramNS  # 修正导入路径

def train_model(training_data, vocab_size):
    """训练模型"""
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = SkipGramNS(vocab_size, Config.EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 准备数据加载器
    targets = np.array([d[0] for d in training_data])
    contexts = np.array([d[1] for d in training_data])
    neg_samples_array = np.array([d[2] for d in training_data])
    
    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(targets),
        torch.LongTensor(contexts),
        torch.LongTensor(neg_samples_array)
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
        
        for target, context, neg_samples in progress:
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
    # 加载训练数据
    training_data = np.load('outputs/training_data.npy', allow_pickle=True)
    
    # 加载词汇表以获取词汇表大小
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    
    # 训练模型
    train_model(training_data, vocab_size)