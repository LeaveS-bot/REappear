# data_preparation.py - 预处理本地文本文件
import os
import re
import sys
from config import Config

def preprocess_text(file_path):
    """文本预处理"""
    print(f"正在处理文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return []
    
    try:
        # 尝试不同编码读取文件
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        text = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                print(f"使用编码 {encoding} 成功读取文件")
                break
            except UnicodeDecodeError:
                print(f"编码 {encoding} 失败")
                continue
        if text is None:
            print("错误: 无法用任何编码读取文件")
            return []
        
        # 清理文本
        print("开始清理文本...")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
        text = re.sub(r'\d+', '', text)      # 移除数字
        
        # 检查清理后的文本
        if not text.strip():
            print("警告: 清理后的文本为空")
            return []
        
        tokens = text.split()
        print(f"生成 {len(tokens)} 个tokens")
        
        # 显示前20个tokens作为示例
        print("前20个tokens示例:", tokens[:20])
        
        return tokens
    except Exception as e:
        print(f"处理文本时发生异常: {str(e)}")
        return []

if __name__ == "__main__":
    print("开始数据准备...")
    print(f"配置文件路径: {os.path.abspath('config.py')}")
    print(f"数据文件路径: {os.path.abspath(Config.DATA_FILE)}")
    
    tokens = preprocess_text(Config.DATA_FILE)
    
    if tokens:
        # 保存处理后的tokens
        os.makedirs('data', exist_ok=True)
        output_path = 'data/preprocessed_tokens.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(' '.join(tokens))
        
        print(f"数据预处理完成！总词数: {len(tokens)}")
        print(f"预处理后的文本已保存到: {output_path}")
    else:
        print("错误: 未能生成有效的 tokens")
        sys.exit(1)  # 退出并返回错误代码