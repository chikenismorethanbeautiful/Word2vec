import jieba
import re

# 定义一个函数来读取中文文本文件并转换为句子列表
def read_chinese_file_to_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取所有文本
        text = file.read()
        # 使用正则表达式清洗文本，去除换行符和其他无关字符
        text = re.sub(r'[^\u4e00-\u9fff\d，。！？、；："\']', '', text)
        text = re.sub(r'\s+', '', text)  # 替换所有空白字符为一个空格
        text = re.sub(r'\n', '', text)  # 替换换行符为一个空格
        # 使用jieba进行中文分词
        words = jieba.cut(text)
        # 将分词结果转换为句子列表
        sentences = [list(words)]
        return sentences
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import Word2Vec

# 文件路径
file_path = 'sanguoyanyi.txt'
# 调用函数并获取句子列表
sentences = read_chinese_file_to_sentences(file_path)
# 训练Word2vec模型
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=100)
# 获取单词的向量
word_vectors = model.wv
# 保存模型，以便后续使用
model.save("sanguo_w2v.model")
loaded_model = Word2Vec.load("sanguo_w2v.model")
# 找出与某个词最相似的词
similar_words = loaded_model.wv.most_similar('刘备')

print("语义相似性:")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

analogy_words = loaded_model.wv.most_similar(
    positive=['刘备', '张飞'], negative=['关羽'], topn=10)

print(f"\n类比推理:刘备 - 张飞 + 关羽: ")
for word, analogy in analogy_words:
    print(f"{word}: {analogy:.4f}")