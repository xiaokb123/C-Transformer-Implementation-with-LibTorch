# test/test.py
import torch
from transformers import Transformer
from datasets.dataset import MyDataset
from datasets.data_utils import make_data
from utils.general_utils import load_model, calculate_bleu
from configs.config import d_model, n_heads, n_layers, dropout, src_vocab_size, tgt_vocab_size, device

def test():
    # 构建词汇表
    src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}
    tgt_vocab = {'P': 0, 'I': 1, 'am': 2, 'a': 3, 'student': 4, 'like': 5, 'to': 6, 'learn': 7, 'male': 8}
    tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载预训练模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    )
    model = load_model(model, "checkpoint_epoch_10.pth")
    model.to(device)
    model.eval()

    # 测试循环
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for enc_input, dec_input, dec_output in loader:
            enc_input, dec_input = enc_input.to(device), dec_input.to(device)

            # 前向传播
            output = model(enc_input, dec_input)
            pred = output.argmax(dim=-1).squeeze().cpu().numpy()

            # 转换为句子
            pred_sentence = [tgt_idx2word[int(idx)] for idx in pred if int(idx) != tgt_vocab['P']]
            ground_truth = [tgt_idx2word[int(idx)] for idx in dec_output.squeeze() if int(idx) != tgt_vocab['P']]

            predictions.append(pred_sentence)
            ground_truths.append(ground_truth)

    # 计算 BLEU 分数
    bleu_score = calculate_bleu(predictions, ground_truths)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    test()