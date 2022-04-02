from raw_model.base_model import BaseModel
from tools import generate_initial_vector, scale_to_unit_length
import random
from collections import defaultdict
import math
import numpy as np


class TransH(BaseModel):
    def __init__(self, kg_dir, model_name="TransH", epochs=1, batch_size=50, dimension=50, learning_rate=0.01,
                 margin=1.0, norm="L1", epsilon=0.9, evaluation_mode="validation"):
        # 储存每个关系对应的头结点预测概率
        super().__init__(kg_dir, model_name, epochs, batch_size, dimension, learning_rate, margin, norm, epsilon,
                         evaluation_mode)
        self.relation_p_dict = dict()

        self._calculate_tph_and_hpt()
        self.relation_normal_embeddings = []
        self.relation_hyper_embeddings = []

    def _init_embeddings(self):
        for _ in range(len(self.kg.entity_ids)):
            self.entity_embeddings.append(generate_initial_vector(self.dimension))
        for _ in range(len(self.kg.relation_ids)):
            self.relation_normal_embeddings.append(scale_to_unit_length(generate_initial_vector(self.dimension)))
            self.relation_hyper_embeddings.append(scale_to_unit_length(generate_initial_vector(self.dimension)))

    def _update_embeddings(self, positive_samples, negative_samples):
        self.total_sample_count += len(positive_samples) * len(negative_samples)

        for positive_sample in positive_samples:
            for negative_sample in negative_samples:

                pos_h = self.entity_embeddings[positive_sample[0]]
                pos_t = self.entity_embeddings[positive_sample[2]]
                r_normal = self.relation_normal_embeddings[positive_sample[1]]
                r_hyper = self.relation_hyper_embeddings[positive_sample[1]]

                neg_h = self.entity_embeddings[negative_sample[0]]
                neg_t = self.entity_embeddings[negative_sample[2]]

                # 计算向量之间的距离
                pos_distance = self._get_distance(pos_h, r_normal, r_hyper, pos_t)
                neg_distance = self._get_distance(neg_h, r_normal, r_hyper, neg_t)

                loss1 = self.margin + pos_distance - neg_distance
                loss2 = _unit_constraint(pos_h, pos_t, neg_h, neg_t)
                if loss1 > 0:
                    self.total_loss += loss1
                    i = np.ones(self.dimension)
                    # 默认是L2范式
                    pos_gradient = 2 * (pos_h - np.dot(r_normal, pos_h) * r_normal +
                                        r_hyper - pos_t +
                                        np.dot(r_normal, pos_t) *
                                        r_normal) * (i - r_normal ** 2)
                    neg_gradient = 2 * (neg_h - np.dot(r_normal, neg_h) * r_normal +
                                        r_hyper - neg_t +
                                        np.dot(r_normal, neg_t) *
                                        r_normal) * (i - r_normal ** 2)
                    normal_gradient = 2 * (pos_h - np.dot(r_normal, pos_h) * r_normal +
                                           r_hyper - pos_t +
                                           np.dot(r_normal, pos_t) *
                                           r_normal) * (pos_t - pos_h) * 2 * r_normal - 2 * (
                                              neg_h - np.dot(r_normal, neg_h) * r_normal +
                                              r_hyper - neg_t +
                                              np.dot(r_normal, neg_t) *
                                              r_normal) * (neg_t - neg_h) * 2 * r_normal
                    hyper_gradient = 2 * (pos_h - np.dot(r_normal, pos_h) * r_normal +
                                          - pos_t + np.dot(r_normal, pos_t)
                                          * r_normal) - 2 * (neg_h - np.dot(r_normal,
                                                                            neg_h) * r_normal +
                                                             - neg_t +
                                                             np.dot(r_normal, neg_t) *
                                                             r_normal)

                    # 如果是L1范式再改变梯度的具体值
                    if self.norm == "L1":
                        for i in range(len(pos_gradient)):
                            if pos_gradient[i] > 0:
                                pos_gradient[i] = 1
                            else:
                                pos_gradient[i] = -1
                            if neg_gradient[i] > 0:
                                neg_gradient[i] = 1
                            else:
                                neg_gradient[i] = -1
                            if normal_gradient[i] > 0:
                                normal_gradient[i] = 1
                            else:
                                normal_gradient[i] = -1
                            if hyper_gradient[i] > 0:
                                hyper_gradient[i] = 1
                            else:
                                hyper_gradient[i] = -1

                    # 损失函数希望达到的理想情况是，正例的d(h + r, t) 尽可能小，负例的d(h' + r', t') 尽可能大，
                    # 这样才能让总体的loss趋向于0。因此在梯度下降过程中，
                    # 正例中h和r逐渐减小，但t逐渐增大； 负例中h'和r'逐渐增大，而t'逐渐减小
                    pos_h -= self.learning_rate * pos_gradient
                    r_normal -= self.learning_rate * normal_gradient
                    r_hyper -= self.learning_rate * hyper_gradient
                    pos_t += self.learning_rate * pos_gradient

                    # 如果负例替换的是头实体, 则正例的尾实体更新两次, 一次满足正例，一次满足负例
                    if positive_sample[0] != negative_sample[0]:
                        neg_h += self.learning_rate * neg_gradient
                        pos_t -= self.learning_rate * neg_gradient
                    # 如果负例替换的是尾实体, 则正例的头实体更新两次
                    elif positive_sample[2] != negative_sample[2]:
                        neg_t -= self.learning_rate * neg_gradient
                        pos_h += self.learning_rate * neg_gradient

                    # 将正例头尾实体新的向量表示放缩到单位长度, 并替换原来的向量表示
                    self.entity_embeddings[positive_sample[0]] = scale_to_unit_length(pos_h)
                    self.entity_embeddings[positive_sample[2]] = scale_to_unit_length(pos_t)
                    # 将负例中被替换的头实体或尾实体的新向量表示放缩到单位长度, 并替换原来的向量表示
                    if positive_sample[0] != negative_sample[0]:
                        self.entity_embeddings[negative_sample[0]] = scale_to_unit_length(neg_h)
                    elif positive_sample[2] != negative_sample[2]:
                        self.entity_embeddings[negative_sample[2]] = scale_to_unit_length(neg_t)

                    self.relation_normal_embeddings[positive_sample[1]] = scale_to_unit_length(r_normal)
                    self.relation_hyper_embeddings[positive_sample[1]] = scale_to_unit_length(r_hyper)

    def _generate_pos_neg_batch(self, batch_size):
        """
        这里关于p的说明 tph 表示每一个头结对应的平均尾节点数 hpt 表示每一个尾节点对应的平均头结点数
        当tph > hpt 时 更倾向于替换头 反之则跟倾向于替换尾实体
        举例说明
        在一个知识图谱中，一共有10个实体 和n个关系，如果其中一个关系使两个头实体对应五个尾实体，
        那么这些头实体的平均 tph为2.5，而这些尾实体的平均 hpt只有0.4，
        则此时我们更倾向于替换头实体，
        因为替换头实体才会有更高概率获得正假三元组，如果替换头实体，获得正假三元组的概率为 8/9 而替换尾实体获得正假三元组的概率只有 5/9
        """
        positive_batch = random.sample(self.kg.train_quads, batch_size)
        negative_batch = []

        for (h, r, t, d) in positive_batch:
            random_choice = np.random.random()
            p = self.relation_p_dict[r]
            while True:
                if random_choice < p:
                    h = random.choice(self.kg.entity_ids)
                else:
                    t = random.choice(self.kg.entity_ids)
                if (h, r, t, d) not in self.kg.train_quads_set:
                    break
            negative_batch.append((h, r, t, d))
        return positive_batch, negative_batch

    def _calculate_tph_and_hpt(self):
        r_heads_dict = defaultdict(set)
        r_tails_dict = defaultdict(set)
        for (h, r, t, d) in self.kg.train_quads:
            r_heads_dict[r].add(h)
            r_tails_dict[r].add(t)

        for r in self.kg.relation_ids:
            head_count = len(r_heads_dict[r])
            tail_count = len(r_tails_dict[r])
            # print("r: %d, head_count: %d, tail_count: %d" % (r, head_count, tail_count))
            tph = head_count / tail_count
            hpt = tail_count / head_count
            # print(tph / (tph + hpt))
            self.relation_p_dict[r] = tph / (tph + hpt)

    def _get_distance(self, h, r_normal, r_hyper, t):
        if self.norm == "L1":
            result = np.sum(
                np.abs(h - np.dot(r_normal, h) * r_normal + r_hyper - t + np.dot(r_normal, t) * r_normal))
        elif self.norm == "L2":
            result = np.sum(
                np.square(h - np.dot(r_normal, h) * r_normal + r_hyper - t + np.dot(r_normal, t) * r_normal))
        else:
            raise RuntimeError("wrong norm")

        return result


# 其他实现说正交约束影响很小, 所以只报保留模长约束，舍弃正交约束
def _unit_constraint(pos_h, pos_t, neg_h, neg_t):
    return np.linalg.norm(pos_h) ** 2 - 1 + np.linalg.norm(pos_t) ** 2 - 1 + np.linalg.norm(
        neg_h) ** 2 - 1 + np.linalg.norm(
        neg_t) ** 2 - 1
