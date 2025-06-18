# evaluate_model.py - 评估词向量
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import config
import os

def find_similar_words(word, embeddings, vocab, idx_to_word):
    """查找相似词"""
    if word not in vocab:
        print(f"单词 '{word}' 不在词汇表中。")
        return []
    
    word_id = vocab[word]
    word_vec = embeddings[word_id].reshape(1, -1)
    sims = cosine_similarity(word_vec, embeddings)[0]
    similar_ids = np.argsort(sims)[::-1][1:config.Config.TOP_N+1]  # 排除自身
    
    return [(idx_to_word[idx], sims[idx]) for idx in similar_ids]

def visualize_embeddings(embeddings, vocab, idx_to_word):
    """可视化词向量"""
    # 选择最常用的词
    common_words = [word for word, _ in Counter(list(vocab.keys())).most_common(config.Config.NUM_WORDS_VISUALIZE) 
                   if word != '<unk>']
    word_ids = [vocab[word] for word in common_words]
    
    # 降维
    tsne = TSNE(n_components=2, random_state=0, perplexity=15)
    embeddings_2d = tsne.fit_transform(embeddings[word_ids])
    
    # 绘图
    plt.figure(figsize=(15, 15))
    for i, word in enumerate(common_words):
        x, y = embeddings_2d[i, :]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), alpha=0.7, fontsize=9)
    
    plt.title('Word2Vec 词向量可视化 (t-SNE)')
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/word_embeddings_visualization.png', dpi=300)
    plt.close()
    print("词向量可视化已保存到 outputs/word_embeddings_visualization.png")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 加载词向量
    embeddings = np.load('outputs/word_embeddings.npy')
    
    # 加载词汇表和反向映射
    with open('outputs/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('outputs/idx_to_word.pkl', 'rb') as f:
        idx_to_word = pickle.load(f)
    
    # 测试相似词
    test_words = ['king', 'queen', 'man', 'woman', 'london', 'france', 'romeo', 'juliet']
    for word in test_words:
        similar = find_similar_words(word, embeddings, vocab, idx_to_word)
        if similar:
            print(f"\n与 '{word}' 相似的词:")
            for w, sim in similar:
                print(f"{w}: {sim:.4f}")
    
    # 可视化
    visualize_embeddings(embeddings, vocab, idx_to_word)