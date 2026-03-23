import jieba
import re
import time
import random
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier

# ==================== 自定义词典：确保人物和绰号被整体分词 ====================
character_nickname = {
    '宋江': '及时雨', '卢俊义': '玉麒麟', '吴用': '智多星', '公孙胜': '入云龙',
    '关胜': '大刀', '林冲': '豹子头', '秦明': '霹雳火', '呼延灼': '双鞭',
    '花荣': '小李广', '柴进': '小旋风', '李应': '扑天雕', '朱仝': '美髯公',
    '鲁智深': '花和尚', '武松': '行者', '董平': '双枪将', '张清': '没羽箭',
    '杨志': '青面兽', '徐宁': '金枪手', '索超': '急先锋', '戴宗': '神行太保',
    '刘唐': '赤发鬼', '李逵': '黑旋风', '史进': '九纹龙', '穆弘': '没遮拦',
    '雷横': '插翅虎', '李俊': '混江龙', '阮小二': '立地太岁', '张横': '船火儿',
    '阮小五': '短命二郎', '张顺': '浪里白条', '阮小七': '活阎罗', '杨雄': '病关索',
    '石秀': '拼命三郎', '解珍': '两头蛇', '解宝': '双尾蝎', '燕青': '浪子',
    '王英': '矮脚虎', '扈三娘': '一丈青', '樊瑞': '混世魔王', '鲍旭': '丧门神',
    '焦挺': '没面目', '李衮': '飞天大圣', '项充': '八臂哪吒', '时迁': '鼓上蚤',
    '段景住': '金毛犬', '萧让': '圣手书生', '裴宣': '铁面孔目', '邓飞': '火眼狻猊',
    '燕顺': '锦毛虎', '杨林': '锦豹子', '凌振': '轰天雷', '蒋敬': '神算子',
    '吕方': '小温侯', '郭盛': '赛仁贵', '安道全': '神医', '皇甫端': '紫髯伯',
    '王定六': '活闪婆', '郁保四': '险道神', '白胜': '白日鼠',
}

for name, nick in character_nickname.items():
    jieba.add_word(name)
    jieba.add_word(nick)

# ==================== 4.1 数据预处理 ====================
def load_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r'[^\u4e00-\u9fff，。！？、；：“”‘’]', '', text)
    raw_sentences = re.split(r'[。！？]', text)
    tokenized_sentences = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = jieba.lcut(sent)
        words = [w for w in words if w.strip() and not re.match(r'^[，。！？、；：“”‘’]$', w)]
        if words:
            tokenized_sentences.append(words)
    return tokenized_sentences

file_path = 'shuihu.txt'
sentences = load_and_preprocess(file_path)
print(f"预处理完成，共 {len(sentences)} 个句子。")

# ==================== 4.2 参数调优记录 ====================
param_grid = [
    {'vector_size': 150, 'window': 5, 'min_count': 1, 'epochs': 200, 'sg': 1},
    {'vector_size': 150, 'window': 8, 'min_count': 2, 'epochs': 200, 'sg': 1},
    {'vector_size': 200, 'window': 5, 'min_count': 1, 'epochs': 200, 'sg': 1},
    {'vector_size': 200, 'window': 8, 'min_count': 2, 'epochs': 200, 'sg': 1},
]

# ==================== 4.3 评价指标应用 ====================
similar_pairs = [
    ('宋江', '卢俊义'),
    ('武松', '鲁智深'),
    ('林冲', '杨志'),
    ('李逵', '张顺'),
]

def generate_analogies(char_dict, model, num=100):
    vocab = set(model.wv.index_to_key)
    valid_pairs = [(char, nick) for char, nick in char_dict.items()
                   if char in vocab and nick in vocab]
    if len(valid_pairs) < 2:
        return []
    analogies = []
    for i in range(len(valid_pairs)):
        a, b = valid_pairs[i]
        for j in range(len(valid_pairs)):
            if i == j:
                continue
            c, d = valid_pairs[j]
            if b != d:
                analogies.append((a, b, c, d))
    random.shuffle(analogies)
    return analogies[:min(num, len(analogies))]

def evaluate_analogies(model, analogies, topk=3):
    correct = 0
    total = 0
    for (a, b, c, d) in analogies:
        try:
            results = model.wv.most_similar(positive=[b, c], negative=[a], topn=topk)
            predicted = [word for word, _ in results]
            if d in predicted:
                correct += 1
            total += 1
        except KeyError:
            continue
    return correct / total if total > 0 else 0, total

# ==================== 执行多组实验 ====================
results = []
for idx, params in enumerate(param_grid, 1):
    print(f"\n开始实验 {idx}：{params}")
    start_time = time.time()
    model = Word2Vec(sentences,
                     vector_size=params['vector_size'],
                     window=params['window'],
                     min_count=params['min_count'],
                     workers=4,
                     epochs=params['epochs'],
                     sg=params['sg'],
                     compute_loss=True)   # 开启损失计算
    train_time = time.time() - start_time

    # 获取训练损失（总累积损失）
    loss = model.get_latest_training_loss()

    # 生成类比
    analogies = generate_analogies(character_nickname, model, num=100)
    analogy_count = len(analogies)
    print(f"生成有效类比数量：{analogy_count}")

    if analogy_count > 0:
        top1_acc, _ = evaluate_analogies(model, analogies, topk=1)
        top3_acc, _ = evaluate_analogies(model, analogies, topk=3)
    else:
        top1_acc = top3_acc = 0.0

    # 计算语义相似度平均值
    sim_scores = []
    for w1, w2 in similar_pairs:
        if w1 in model.wv and w2 in model.wv:
            sim_scores.append(model.wv.similarity(w1, w2))
    avg_sim = np.mean(sim_scores) if sim_scores else 0

    results.append({
        'batch': idx,
        'vector_size': params['vector_size'],
        'window': params['window'],
        'min_count': params['min_count'],
        'epochs': params['epochs'],
        'sg': params['sg'],
        'train_time': f"{train_time:.2f}s",
        'loss': f"{loss:.2e}",                # 科学计数法，避免数值过大
        'avg_similarity': f"{avg_sim:.4f}",
        'top1_analogy_acc': f"{top1_acc:.2%}",
        'top3_analogy_acc': f"{top3_acc:.2%}",
        'analogy_count': analogy_count
    })

# ==================== 输出参数调优记录表格 ====================
print("\n" + "=" * 90)
print("参数调优记录（扩展实验4.2）")
print("=" * 90)
print("| 实验批次 | vector_size | window | min_count | epochs |  sg  | 训练耗时 | 训练损失   | 平均语义相似度 | 有效类比数 | Top1准确率 | Top3准确率 |")
print("| :------: | :---------: | :----: | :-------: | :----: | :--: | :------: | :--------: | :------------: | :--------: | :--------: | :--------: |")
for res in results:
    print(f"|    {res['batch']:2d}    |     {res['vector_size']:3d}     |   {res['window']:2d}   |     {res['min_count']:2d}     |   {res['epochs']:3d}  | {'CBOW' if res['sg'] == 0 else 'SG'} | {res['train_time']:>7s} | {res['loss']:>9s} |      {res['avg_similarity']}      |    {res['analogy_count']:3d}    |    {res['top1_analogy_acc']}    |    {res['top3_analogy_acc']}    |")

# 选出最佳模型（以Top3准确率为准）
best = max(results, key=lambda x: (float(x['top3_analogy_acc'].strip('%')), x['analogy_count']))
print(f"\n最佳模型为实验 {best['batch']}，Top3准确率 {best['top3_analogy_acc']}（有效类比 {best['analogy_count']} 个）")

# ==================== 外在评估（使用最佳模型）====================
# 重新训练最佳模型（或直接用之前训练好的，这里重新训练确保一致性）
best_params = param_grid[best['batch'] - 1]
model_best = Word2Vec(sentences,
                      vector_size=best_params['vector_size'],
                      window=best_params['window'],
                      min_count=best_params['min_count'],
                      workers=4,
                      epochs=best_params['epochs'],
                      sg=best_params['sg'])

# ----- 1. 聚类轮廓系数 -----
person_list = [name for name in character_nickname.keys() if name in model_best.wv]
vectors = np.array([model_best.wv[name] for name in person_list])

# 尝试不同k值，选择轮廓系数最大的k（肘部法则可选，这里直接取k=6作为示例）
# 但为了客观，可以遍历k=2~10，选择最佳轮廓系数
sil_scores = []
k_range = range(2, min(11, len(person_list)))
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vectors)
    sil = silhouette_score(vectors, labels)
    sil_scores.append(sil)
best_k = k_range[np.argmax(sil_scores)]
best_sil = max(sil_scores)

print(f"\n外在评估1：人物词向量聚类轮廓系数（最佳k={best_k}）：{best_sil:.4f}")

# ----- 2. 下游任务：天罡/地煞二分类 F1-score 对比 -----
# 天罡星36人名单（来自水浒传）
tiangang_list = [
    '宋江', '卢俊义', '吴用', '公孙胜', '关胜', '林冲', '秦明', '呼延灼',
    '花荣', '柴进', '李应', '朱仝', '鲁智深', '武松', '董平', '张清',
    '杨志', '徐宁', '索超', '戴宗', '刘唐', '李逵', '史进', '穆弘',
    '雷横', '李俊', '阮小二', '张横', '阮小五', '张顺', '阮小七', '杨雄',
    '石秀', '解珍', '解宝', '燕青'
]

# 地煞星72人，这里仅取在character_nickname中且在词表中的人物作为地煞样本
disha_list = [name for name in character_nickname.keys() if name not in tiangang_list]

# 构建数据集：使用在词表中且属于上述两类的所有人物
X_names = []
y = []
for name in tiangang_list:
    if name in model_best.wv:
        X_names.append(name)
        y.append(1)   # 天罡为1
for name in disha_list:
    if name in model_best.wv:
        X_names.append(name)
        y.append(0)   # 地煞为0

X = np.array([model_best.wv[name] for name in X_names])
y = np.array(y)

if len(X) > 10:  # 确保有足够样本
    # 基线模型：多数类预测
    dummy = DummyClassifier(strategy='most_frequent')
    dummy_scores = cross_val_score(dummy, X, y, cv=StratifiedKFold(5), scoring='f1_macro')
    baseline_f1 = np.mean(dummy_scores)

    # 逻辑回归分类器
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=StratifiedKFold(5), scoring='f1_macro')
    model_f1 = np.mean(lr_scores)

    f1_improve = (model_f1 - baseline_f1) / baseline_f1 * 100
    print(f"外在评估2：天罡/地煞分类 F1-score（基线）: {baseline_f1:.4f}")
    print(f"外在评估2：天罡/地煞分类 F1-score（逻辑回归）: {model_f1:.4f}")
    print(f"F1-score 提升: {f1_improve:.2f}%")
else:
    print("外在评估2：天罡/地煞样本不足，无法进行下游任务。")
    baseline_f1 = model_f1 = f1_improve = 0.0