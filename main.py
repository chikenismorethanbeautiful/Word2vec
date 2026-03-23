import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import jieba
import numpy as np
import re
from collections import Counter
import time
import os

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_preprocess_data(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', line)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) > 5:
                sentences.append(cleaned)
    return sentences


def chinese_tokenize(sentences):
    tokenized_sentences = []
    for i, sentence in enumerate(sentences):
        words = jieba.lcut(sentence)
        words = [word.strip() for word in words if len(word.strip()) > 1]
        tokenized_sentences.append(words)
        if i % 200 == 0:
            print(f"已处理 {i}/{len(sentences)} 条数据")
    return tokenized_sentences


def analyze_vocabulary(tokenized_corpus):
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_freq = Counter(all_words)
    print(f"总词汇量: {len(all_words)}")
    print(f"唯一词汇数: {len(word_freq)}")
    print(f"平均句子长度: {np.mean([len(sentence) for sentence in tokenized_corpus]):.2f}")
    return word_freq


def build_vocab(tokenized_corpus, min_count=5):
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}
    idx_to_word = ['<PAD>', '<UNK>'] + list(vocab.keys())
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    print(f"词汇表大小: {len(word_to_idx)}")
    return word_to_idx, idx_to_word, vocab


def create_training_data(tokenized_corpus, word_to_idx, window_size=5):
    training_data = []
    vocab_size = len(word_to_idx)
    unk_idx = word_to_idx.get('<UNK>', 0)
    word_counts = np.zeros(vocab_size)
    for word, idx in word_to_idx.items():
        if word in vocab:
            word_counts[idx] = vocab[word]
    word_distribution = np.power(word_counts, 0.75)
    word_distribution = word_distribution / word_distribution.sum()
    for sentence in tokenized_corpus:
        sentence_indices = [word_to_idx.get(word, unk_idx) for word in sentence]
        for i, target_idx in enumerate(sentence_indices):
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)
            for j in range(start, end):
                if j != i:
                    context_idx = sentence_indices[j]
                    training_data.append((target_idx, context_idx))
    print(f"创建了 {len(training_data)} 个正样本对")
    return training_data, word_distribution


def precompute_negatives_fast(training_data, word_distribution, num_negatives=5):
    print("预生成负样本（快速矢量化）...")
    vocab_size = len(word_distribution)
    targets = np.array([t for t, c in training_data], dtype=np.int32)
    contexts = np.array([c for t, c in training_data], dtype=np.int32)
    negatives = np.random.choice(vocab_size, size=(len(training_data), num_negatives), p=word_distribution)
    targets_t = torch.tensor(targets, dtype=torch.long)
    contexts_t = torch.tensor(contexts, dtype=torch.long)
    negs_t = torch.tensor(negatives, dtype=torch.long)
    return targets_t, contexts_t, negs_t


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50):
        super(Word2VecModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        init_range = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word, context_word, negative_words):
        target_embed = self.target_embeddings(target_word)
        context_embed = self.context_embeddings(context_word)
        negative_embed = self.context_embeddings(negative_words)
        positive_score = torch.sum(target_embed * context_embed, dim=1)
        positive_score = torch.clamp(positive_score, max=10, min=-10)
        target_embed_expanded = target_embed.unsqueeze(1)
        negative_score = torch.bmm(negative_embed, target_embed_expanded.transpose(1, 2))
        negative_score = torch.clamp(negative_score.squeeze(2), max=10, min=-10)
        return positive_score, negative_score


def skipgram_loss(positive_score, negative_score):
    positive_loss = -torch.log(torch.sigmoid(positive_score))
    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)
    return (positive_loss + negative_loss).mean()


def train_word2vec_fast(model, dataset, batch_size=32768, epochs=5, lr=0.01, log_file='loss_log.txt'):
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n开始快速训练，批量大小: {batch_size}")
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for batch_idx, (targets, contexts, negatives) in enumerate(dataloader):
            targets = targets.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pos_score, neg_score = model(targets, contexts, negatives)
                loss = skipgram_loss(pos_score, neg_score)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if batch_idx % 500 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1} 完成 | 平均损失: {avg_loss:.4f} | 耗时: {time.time() - start_time:.2f}s")

    # 保存损失记录
    with open(log_file, 'w', encoding='utf-8') as f:
        for i, loss in enumerate(epoch_losses):
            f.write(f"Epoch {i + 1}: {loss:.4f}\n")
    print(f"训练损失已保存至 {log_file}")
    return model


def get_word_vectors(model, word_to_idx):
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(word_to_idx)).to(device)
        word_vectors = model.target_embeddings(all_indices).detach().cpu()
    word_vectors_dict = {}
    for word, idx in word_to_idx.items():
        word_vectors_dict[word] = word_vectors[idx]
    return word_vectors_dict, word_vectors


class PyTorchWord2VecWrapper:
    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            self.vectors_dict = word_vectors_dict
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            self.vectors = np.stack(list(word_vectors_dict.values()))

        def __getitem__(self, word):
            return self.vectors_dict.get(word, None)

        def __contains__(self, word):
            return word in self.vectors_dict

        def similarity(self, word1, word2):
            if word1 not in self.vectors_dict or word2 not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word1} 或 {word2}")
            vec1 = self.vectors_dict[word1]
            vec2 = self.vectors_dict[word2]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)

        def most_similar(self, word, topn=10):
            if word not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word}")
            target_vec = self.vectors_dict[word]
            similarities = []
            for w, vec in self.vectors_dict.items():
                if w == word:
                    continue
                norm_target = np.linalg.norm(target_vec)
                norm_vec = np.linalg.norm(vec)
                if norm_target == 0 or norm_vec == 0:
                    sim = 0.0
                else:
                    sim = np.dot(target_vec, vec) / (norm_target * norm_vec)
                similarities.append((w, sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:topn]


def word_analogy(a, b, c, model, topn=5):
    if a not in model.wv or b not in model.wv or c not in model.wv:
        raise KeyError("其中一个词不在词汇表中")
    vec_a = model.wv[a]
    vec_b = model.wv[b]
    vec_c = model.wv[c]
    target_vec = vec_b - vec_a + vec_c
    all_vectors = model.wv.vectors
    norms = np.linalg.norm(all_vectors, axis=1, keepdims=True)
    normalized = all_vectors / norms
    target_norm = np.linalg.norm(target_vec)
    target_normalized = target_vec / target_norm if target_norm != 0 else target_vec
    sims = np.dot(normalized, target_normalized)
    exclude = {model.wv.word_to_idx[a], model.wv.word_to_idx[b], model.wv.word_to_idx[c]}
    sorted_idx = np.argsort(sims)[::-1]
    results = []
    for idx in sorted_idx:
        if idx not in exclude:
            results.append((model.wv.idx_to_word[idx], sims[idx]))
            if len(results) >= topn:
                break
    return results


def extended_evaluation(w2v_model, output_file='evaluation_results.txt'):
    """扩展评估：计算领域词对平均相似度、相似词领域相关性等，并保存结果"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("词向量模型扩展评估结果\n")
        f.write("=" * 50 + "\n\n")

        # 1. 相似词查找（早泄）
        f.write("【指标1】与“早泄”最相似的10个词：\n")
        if '早泄' in w2v_model.wv:
            sim_words = w2v_model.wv.most_similar('早泄', topn=10)
            for word, score in sim_words:
                f.write(f"  {word}: {score:.3f}\n")
            f.write("\n")
        else:
            f.write("  '早泄'不在词汇表中\n\n")

        # 2. 领域相关词对相似度及平均
        f.write("【指标2】领域相关词对相似度：\n")
        pairs = [('早泄', '射精'), ('前列腺', '炎症'), ('病因', '诱因'),
                 ('手淫', '自慰'), ('糖尿病', '血糖'), ('心理', '焦虑')]
        similarities = []
        for w1, w2 in pairs:
            if w1 in w2v_model.wv and w2 in w2v_model.wv:
                sim = w2v_model.wv.similarity(w1, w2)
                similarities.append(sim)
                f.write(f"  '{w1}' 与 '{w2}': {sim:.3f}\n")
            else:
                f.write(f"  '{w1}' 或 '{w2}' 不在词汇表中\n")
        if similarities:
            avg_sim = np.mean(similarities)
            f.write(f"\n  平均相似度: {avg_sim:.3f}\n\n")
        else:
            f.write("\n")

        # 3. 类比推理测试
        f.write("【指标3】类比推理测试：\n")
        analogies = [('早泄', '病因', '前列腺'), ('早泄', '症状', '射精'), ('治疗', '药物', '手术')]
        for a, b, c in analogies:
            try:
                res = word_analogy(a, b, c, w2v_model, topn=3)
                f.write(f"  '{a}' : '{b}' 如同 '{c}' : ?\n")
                for w, s in res:
                    f.write(f"    -> {w}: {s:.3f}\n")
            except KeyError as e:
                f.write(f"  无法完成推理: {e}\n")
        f.write("\n")

        # 4. 相似词领域相关性统计（以“早泄”为例）
        f.write("【指标4】相似词领域相关性统计（以“早泄”为例）：\n")
        if '早泄' in w2v_model.wv:
            # 预定义一些领域关键词（可根据实际情况扩充）
            domain_keywords = {'早泄', '阳痿', '射精', '前列腺', '炎症', '病因', '诱因',
                               '手淫', '自慰', '糖尿病', '血糖', '心理', '焦虑', '治疗',
                               '药物', '手术', '症状', '疼痛', 'B超', '清创', '显微外科'}
            sim_words = w2v_model.wv.most_similar('早泄', topn=10)
            domain_count = sum(1 for w, _ in sim_words if w in domain_keywords)
            ratio = domain_count / len(sim_words)
            f.write(f"  Top10中领域相关词数量: {domain_count}/10, 比例: {ratio:.1%}\n")
            for w, _ in sim_words:
                f.write(f"    {w}\n")
        else:
            f.write("  '早泄'不在词汇表中\n")

    print(f"扩展评估结果已保存至 {output_file}")


if __name__ == '__main__':
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")

    file_path = "./男科5-13000.txt"
    sentences = load_and_preprocess_data(file_path)
    print(f"总共加载了 {len(sentences)} 条文本数据")

    tokenized_corpus = chinese_tokenize(sentences)
    print("分词完成！")

    word_frequency = analyze_vocabulary(tokenized_corpus)

    word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=5)

    training_data, word_distribution = create_training_data(tokenized_corpus, word_to_idx, window_size=5)

    targets_t, contexts_t, negs_t = precompute_negatives_fast(training_data, word_distribution, num_negatives=5)
    dataset = TensorDataset(targets_t, contexts_t, negs_t)

    model = Word2VecModel(vocab_size=len(word_to_idx), embedding_dim=50)
    trained_model = train_word2vec_fast(model, dataset, batch_size=32768, epochs=5, lr=0.01, log_file='loss_log.txt')

    word_vectors_dict, all_vectors = get_word_vectors(trained_model, word_to_idx)
    torch.save(word_vectors_dict, "word_vectors.pt")
    print("词向量已保存为 word_vectors.pt")

    w2v_model = PyTorchWord2VecWrapper(word_vectors_dict, word_to_idx, idx_to_word)

    print("\n" + "=" * 50)
    print("词向量模型评价（领域适配）")
    print("=" * 50)

    # 原控制台输出保持不变
    print("\n【指标1】相似词查找（与“早泄”最相似的词）：")
    if '早泄' in w2v_model.wv:
        for word, score in w2v_model.wv.most_similar('早泄', topn=10):
            print(f"  {word}: {score:.3f}")
    else:
        print("  '早泄'不在词汇表中")

    print("\n【指标2】领域相关词对相似度：")
    pairs = [('早泄', '射精'), ('前列腺', '炎症'), ('病因', '诱因'), ('手淫', '自慰'), ('糖尿病', '血糖'),
             ('心理', '焦虑')]
    for w1, w2 in pairs:
        if w1 in w2v_model.wv and w2 in w2v_model.wv:
            print(f"  '{w1}' 与 '{w2}': {w2v_model.wv.similarity(w1, w2):.3f}")
        else:
            print(f"  '{w1}' 或 '{w2}' 不在词汇表中")

    print("\n【指标3】类比推理测试：")
    analogies = [('早泄', '病因', '前列腺'), ('早泄', '症状', '射精'), ('治疗', '药物', '手术')]
    for a, b, c in analogies:
        try:
            res = word_analogy(a, b, c, w2v_model, topn=3)
            print(f"  '{a}' : '{b}' 如同 '{c}' : ?")
            for w, s in res:
                print(f"    -> {w}: {s:.3f}")
        except KeyError as e:
            print(f"  无法完成推理: {e}")

    extended_evaluation(w2v_model, output_file='evaluation_results.txt')