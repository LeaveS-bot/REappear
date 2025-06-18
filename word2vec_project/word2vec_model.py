# word2vec_model.py - 定义Skip-Gram模型
import torch
import torch.nn as nn

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNS, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward(self, target, context, neg_samples):
        # 正样本
        emb_target = self.in_embed(target)  # (batch_size, emb_dim)
        emb_context = self.out_embed(context)  # (batch_size, emb_dim)
        pos_score = torch.sum(emb_target * emb_context, dim=1)  # (batch_size)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-7)  # 正样本损失，添加小值防止log(0)
        
        # 负样本
        neg_emb = self.out_embed(neg_samples)  # (batch_size, num_neg, emb_dim)
        emb_target_expanded = emb_target.unsqueeze(1)  # (batch_size, 1, emb_dim)
        neg_score = torch.sum(emb_target_expanded * neg_emb, dim=2)  # (batch_size, num_neg)
        neg_loss = -torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-7), dim=1)  # 负样本损失
        
        return torch.mean(pos_loss + neg_loss)
    
    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()