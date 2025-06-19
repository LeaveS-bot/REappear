# config.py - 统一管理所有参数

class Config:
    # 数据参数 - 使用本地文件
    DATA_FILE = '/home/sfr/REappear/word2vec_project/data/Temp1.txt'  # 你的本地文本文件路径
    MIN_COUNT = 5  # 词汇表最小词频
    
    # 训练数据生成参数
    WINDOW_SIZE = 3
    NUM_NEG_SAMPLES = 5
    
    # 模型参数
    EMBEDDING_DIM = 100
    
    # 训练参数
    BATCH_SIZE = 256  # 减小以节省内存
    NUM_EPOCHS = 50    # 减少训练轮数
    LEARNING_RATE = 0.01
    
    # 评估参数
    TOP_N = 10  # 查找相似词的数量
    NUM_WORDS_VISUALIZE = 100  # 可视化词的数量