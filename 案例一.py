
from gensim.models import KeyedVectors

# 加载预训练模型
pretrained_model_path = 'GoogleNews-vectors-negative300.bin.gz'
model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)

# 计算词语的相似性
word1 = 'cat'
word2 = 'dog'
similarity = model.similarity(word1, word2)
print(f"The similarity between '{word1}' and '{word2}' is: {similarity:.4f}")

word_a = 'king'
word_b = 'man'
word_c = 'woman'
result = model.most_similar(positive=[word_b, word_c], negative=[word_a])
print(f"The result for the analogy '{word_a} : {word_b} :: {word_c} : ' is {result[0][0]}")