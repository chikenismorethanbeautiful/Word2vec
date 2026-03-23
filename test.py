import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import jieba
import re

# ==================== 1. 加载已训练的词向量 ====================
word_vectors_dict = torch.load('word_vectors.pt', map_location='cpu')

# 重建词汇表（与训练时保持一致）
# 注意：word_vectors_dict 中不包含 <PAD> 和 <UNK>，需手动补充
idx_to_word = ['<PAD>', '<UNK>'] + list(word_vectors_dict.keys())
word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

# 将 Tensor 转换为 numpy 数组，以便后续计算
word_vectors_numpy = {word: vec.numpy() for word, vec in word_vectors_dict.items()}

# ==================== 2. 包装类（兼容gensim风格） ====================
class PyTorchWord2VecWrapper:
    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            # 存储为 numpy 数组
            self.vectors_dict = {word: vec.numpy() for word, vec in word_vectors_dict.items()}
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            self.vectors = np.stack(list(self.vectors_dict.values()))

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

# 初始化包装类
w2v_model = PyTorchWord2VecWrapper(word_vectors_dict, word_to_idx, idx_to_word)
print("词向量加载完成，维度:", w2v_model.vector_size)

# ==================== 3. 内在评估（来自 evaluation_results.txt） ====================
print("\n" + "="*50)
print("内在评估结果（来自扩展评估文件）")
print("="*50)

# 指标1：与“早泄”最相似的10个词
print("\n【指标1】与“早泄”最相似的10个词：")
if '早泄' in w2v_model.wv:
    for word, score in w2v_model.wv.most_similar('早泄', topn=10):
        print(f"  {word}: {score:.3f}")

# 指标2：领域相关词对相似度及平均
print("\n【指标2】领域相关词对相似度：")
pairs = [('早泄','射精'),('前列腺','炎症'),('病因','诱因'),
         ('手淫','自慰'),('糖尿病','血糖'),('心理','焦虑')]
similarities = []
for w1, w2 in pairs:
    if w1 in w2v_model.wv and w2 in w2v_model.wv:
        sim = w2v_model.wv.similarity(w1, w2)
        similarities.append(sim)
        print(f"  '{w1}' 与 '{w2}': {sim:.3f}")
    else:
        print(f"  '{w1}' 或 '{w2}' 不在词汇表中")
if similarities:
    avg_sim = np.mean(similarities)
    print(f"\n  平均相似度: {avg_sim:.3f}")

# 指标3：类比推理测试
print("\n【指标3】类比推理测试：")
analogies = [('早泄','病因','前列腺'), ('早泄','症状','射精'), ('治疗','药物','手术')]
for a, b, c in analogies:
    try:
        res = word_analogy(a, b, c, w2v_model, topn=3)
        print(f"  '{a}' : '{b}' 如同 '{c}' : ?")
        for w, s in res:
            print(f"    -> {w}: {s:.3f}")
    except KeyError as e:
        print(f"  无法完成推理: {e}")

# 指标4：相似词领域相关性统计（以“早泄”为例）
print("\n【指标4】相似词领域相关性统计（以“早泄”为例）：")
if '早泄' in w2v_model.wv:
    # 预定义领域关键词集（可根据需要扩展）
    domain_keywords = {'早泄','阳痿','射精','前列腺','炎症','病因','诱因',
                       '手淫','自慰','糖尿病','血糖','心理','焦虑','治疗',
                       '药物','手术','症状','疼痛','B超','清创','显微外科',
                       '早射','过快','不坚','早泻','早射要','快用','快后','短是'}
    sim_words = w2v_model.wv.most_similar('早泄', topn=10)
    domain_count = sum(1 for w, _ in sim_words if w in domain_keywords)
    ratio = domain_count / len(sim_words)
    print(f"  Top10中领域相关词数量: {domain_count}/10, 比例: {ratio:.1%}")
    for w, _ in sim_words:
        print(f"    {w}")

# ==================== 4. 外在评估 ====================
print("\n" + "="*50)
print("外在评估结果")
print("="*50)

# ---- 4.1 下游任务F1-score（文本分类示例） ----
# 提示：此处使用模拟数据，请替换为真实标注数据集以获得有效指标
print("\n【下游任务F1-score】")
def sentence_to_vector(sentence, model):
    words = jieba.lcut(sentence)
    vectors = []
    for w in words:
        if w in model.wv:
            vectors.append(model.wv[w])
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# 模拟数据（请替换为真实数据！）
sentences = [
    "早泄怎么办",
    "前列腺炎的症状",
    "糖尿病血糖高",
    "心理焦虑怎么缓解",
    "阳痿治疗药物",
    "手术后疼痛"
]
labels = [0, 1, 2, 3, 0, 1]  # 假设类别：0-男科疾病，1-前列腺，2-糖尿病，3-心理

# 提取特征
X = np.array([sentence_to_vector(s, w2v_model) for s in sentences])
y = np.array(labels)

# 划分训练测试集（样本量少，仅作演示）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"  使用模拟数据计算的F1-score (macro): {f1:.4f}")

# ---- 4.2 聚类分析轮廓系数 ----
print("\n【聚类分析轮廓系数】")
# 选择参与聚类的词（全部词汇，或可过滤低频词）
words = list(w2v_model.wv.vectors_dict.keys())
vectors = np.array([w2v_model.wv[w] for w in words])

# 尝试不同聚类数，取轮廓系数最高的（示例中使用k=5）
best_k = 5
best_score = -1
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    # 如果某个簇只有一个样本，轮廓系数可能无法计算，捕获异常
    if len(set(labels)) > 1:
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_score = score
            best_k = k
print(f"  最佳聚类数 k={best_k}，轮廓系数: {best_score:.4f}")