# train/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer import Transformer, TokenEmbedding, PositionalEncoding
from dataset.dataset import MyDataset
from dataset.data_utils import make_data
from util.general_utils import save_model
from config.config import d_model, n_heads, n_layers, dropout, src_vocab_size, tgt_vocab_size, device

def load_model():
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    return model.to(device)

def train():
    # 构建词汇表
    src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}
    tgt_vocab = {'P': 0, 'I': 1, 'am': 2, 'a': 3, 'student': 4, 'like': 5, 'to': 6, 'learn': 7, 'male': 8}

    # 示例句子
    sentences = [
        ("我是学生", "I am a student", "I am a studentP"),
        ("我喜欢学习", "I like to learn", "I like to learnP")
    ]

    # 数据预处理
    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_vocab, tgt_vocab)
    enc_inputs = torch.LongTensor(enc_inputs)
    dec_inputs = torch.LongTensor(dec_inputs)
    dec_outputs = torch.LongTensor(dec_outputs)

    # 数据加载器
    dataset = MyDataset(enc_inputs, dec_inputs, dec_outputs)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型
    model = load_model()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 训练循环
    for epoch in range(10):
        model.train()
        total_loss = 0
        for enc_input, dec_input, dec_output in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            enc_input, dec_input, dec_output = enc_input.to(device), dec_input.to(device), dec_output.to(device)

            # 前向传播
            output = model(enc_input, dec_input)
            loss = criterion(output.view(-1, tgt_vocab_size), dec_output.view(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(loader):.4f}")

        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_model(model, f"checkpoint_epoch_{epoch + 1}.pth")

if __name__ == "__main__":
    train()