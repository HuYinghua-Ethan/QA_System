import os
import json
import jieba
import numpy as np
from bm25 import BM25
from similarity_function import editing_distance, jaccard_distance
from gensim.models import Word2Vec

'''
基于faq知识库和文本匹配算法进行意图识别，完成单轮问答
'''

class QASystem:
    def __init__(self, knowledge_base_path, algo):
        """
        know_base_path: 知识库文件路径
        algo: 根据需求选择不同的算法
        """
        self.load_knowledge_base(knowledge_base_path)
        self.algo = algo
        if algo == "bm25":
            self.load_bm25()
        elif algo == "word2vec":
            self.load_word2vec()
        else:
            # 其余的算法不需要做事先计算
            pass
    
    def load_knowledge_base(self, knowledge_base_path):
        """
        know_base_path: 知识库文件路径
        这个函数的作用是构建一个字典 key: 标准问 value: 相似问
        """
        self.target_to_questions = {}
        with open(knowledge_base_path, encoding='utf-8') as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    def load_bm25(self):
        """
        这个函数的作用是
        1. 构建self.corpus字典  key: 标准问 value: 所有的相似问构成的列表，只是一个列表，里面包含了分好词的相似问
        2. 实例化BM25模型，并将self.corpus传入模型中
        """
        self.corpus = {}
        for target, questions in self.target_to_questions.items():
            self.corpus[target] = []
            for question in questions:
                self.corpus[target] += jieba.lcut(question)
        self.bm25_model = BM25(self.corpus)
            
    def load_word2vec(self):
        """
        词向量的训练需要一定的时间，如果之前训练过，我们可以直接读取训练好的模型
        注意如果数据集更换了，应当重新训练
        当然，也可以收集一份大量的通用语料，训练一个通用的词向量模型。一般少量的数据来训练效果不会太理想
        """
        if os.path.isfile("model.w2v"):
            self.w2v_model = Word2Vec.load("model.w2v")
        else:
            # 训练语料的准备，把所有问题分词后连在一起
            corpus = []
            for questions in self.target_to_questions.values():
                for question in questions:
                    corpus.append(jieba.lcut(question))
            """
            调用第三方库训练词向量模型
            min_count: 这个参数指的是最小词频，表示只考虑在语料库中出现次数大于等于1的单词。
            vector_size=100: 这个参数指定生成的词向量的维度为100。
            """
            self.w2v_model = Word2Vec(corpus, size=100, min_count=1)
            # 保存模型
            self.w2v_model.save("model.w2v")
        # 借助词向量模型，将知识库中的问题向量化
        self.target_to_vectors = {}
        for target, questions in self.target_to_questions.items():
            vectors = []
            for question in questions:
                vectors.append(self.sentence_to_vector(question))
            self.target_to_vectors[target] = np.array(vectors)

    def sentence_to_vector(self, sentence):
        """
        sentence: 要进行向量化的句子
        步骤：
        1. 分词: 对句子进行分词
        2. 取出词向量：用词向量模型取出每个词的向量，并求和
        3. 计算平均值：用求和出来的词向量 / 句子的总词数
        4. 归一化: vector / |vector|
        """
        vector = np.zeros(self.w2v_model.vector_size)
        words = jieba.lcut(sentence)
        # 所有词的向量相加求平均，作为句子向量
        count = 0
        for word in words:
            if word in self.w2v_model.wv:
                count += 1
                vector += self.w2v_model.wv[word]
        vector = np.array(vector) / count
        # 文本向量归一化
        vector = vector / np.sqrt(np.sum(np.square(vector)))
        return vector
        
    def query(self, user_query):
        results = []
        if self.algo == "editing_distance":
            for target, questions in self.target_to_questions.items():
                scores = [editing_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "jaccard_distance":
            for target, questions in self.target_to_questions.items():
                scores = [jaccard_distance(question, user_query) for question in questions]
                score = max(scores)
                results.append([target, score])
        elif self.algo == "bm25":
            words = jieba.lcut(user_query)
            results = self.bm25_model.get_scores(words)
        elif self.algo == "word2vec":
            query_vector = self.sentence_to_vector(words)
            for target, vectors in self.target_to_vectors.items():
                cos = query_vector.dot(vectors.transpose())
                print(cos)
                results.append([target, np.mean(cos)])
        else:
            assert "unknown algorithm!!"
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        return sorted_results[:3]


if __name__ == '__main__':
    qas = QASystem("data/train.json", "bm25")
    while True:
        user_query = input("请输入用户问题：")
        print("System is working...")
        res = qas.query(user_query)
        print("System response:", res)

