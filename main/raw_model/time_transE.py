from raw_model.base_model import BaseModel
from collections import defaultdict
from tools import get_distance, scale_to_unit_length
import numpy as np


class TimeTransE(BaseModel):
    def __init__(self, kg_dir, epochs=1, batch_size=50, dimension=50, learning_rate=0.01, margin=1.0, norm="L1",
                 epsilon=0.9,
                 evaluation_mode="validation", k=0.01):
        super().__init__(kg_dir, epochs=epochs, batch_size=batch_size, dimension=dimension, learning_rate=learning_rate,
                         margin=margin, norm=norm, epsilon=epsilon, evaluation_mode=evaluation_mode)
        self.k = k
        self.matrix = np.random.rand(self.dimension, self.dimension)
        # 每个头实体对应的(ri, rj)时序关系对集合
        self.pos_h_r_pairs_dict = defaultdict(set)
        self.neg_h_r_pairs_dict = defaultdict(set)
        self._generate_relation_pairs()

    def _update_embeddings(self, positive_samples, negative_samples):
        self.total_sample_count += len(positive_samples) * len(negative_samples)

        for positive_sample in positive_samples:
            loss1 = 0
            # ================ 损失函数 ===============
            for negative_sample in negative_samples:

                pos_h = self.entity_embeddings[positive_sample[0]]
                pos_t = self.entity_embeddings[positive_sample[2]]
                r = self.relation_embeddings[positive_sample[1]]

                neg_h = self.entity_embeddings[negative_sample[0]]
                neg_t = self.entity_embeddings[negative_sample[2]]

                # 计算向量之间的距离
                pos_distance = get_distance(pos_h + r - pos_t, self.norm)
                neg_distance = get_distance(neg_h + r - neg_t, self.norm)

                loss1 = max(self.margin + pos_distance - neg_distance, 0)
                if loss1 > 0:
                    pos_gradient = 2 * (pos_h + r - pos_t)
                    neg_gradient = 2 * (neg_h + r - neg_t)

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

                    # 损失函数希望达到的理想情况是，正例的d(h + r, t) 尽可能小，负例的d(h' + r', t') 尽可能大，
                    # 这样才能让总体的loss趋向于0。因此在梯度下降过程中，
                    # 正例中h和r逐渐减小，但t逐渐增大； 负例中h'和r'逐渐增大，而t'逐渐减小
                    pos_h -= self.learning_rate * pos_gradient
                    pos_t += self.learning_rate * pos_gradient

                    # 如果负例替换的是头实体, 则正例的尾实体更新两次, 一次满足正例，一次满足负例
                    if positive_sample[0] != negative_sample[0]:
                        neg_h += self.learning_rate * neg_gradient
                        pos_t -= self.learning_rate * neg_gradient
                    # 如果负例替换的是尾实体, 则正例的头实体更新两次
                    elif positive_sample[2] != negative_sample[2]:
                        neg_t -= self.learning_rate * neg_gradient
                        pos_h += self.learning_rate * neg_gradient

                    # 关系永远更新两次
                    r -= self.learning_rate * pos_gradient
                    r += self.learning_rate * neg_gradient

                    # 将正例头尾实体新的向量表示放缩到单位长度, 并替换原来的向量表示
                    self.entity_embeddings[positive_sample[0]] = scale_to_unit_length(pos_h)
                    self.entity_embeddings[positive_sample[2]] = scale_to_unit_length(pos_t)
                    # 将负例中被替换的头实体或尾实体的新向量表示放缩到单位长度, 并替换原来的向量表示
                    if positive_sample[0] != negative_sample[0]:
                        self.entity_embeddings[negative_sample[0]] = scale_to_unit_length(neg_h)
                    elif positive_sample[2] != negative_sample[2]:
                        self.entity_embeddings[negative_sample[2]] = scale_to_unit_length(neg_t)

                    # TransE论文提到关系的向量表示不用缩放到单位长度
                    self.relation_embeddings[positive_sample[1]] = r

            # ===================== 正则化项 =========================
            loss2 = self._regularization(positive_sample)
            self.total_loss += loss1 + loss2

    def _regularization(self, pos_quad):
        h = pos_quad[0]
        r = pos_quad[1]
        pos_r_pairs = self.pos_h_r_pairs_dict[(h, r)]
        neg_r_pairs = self.neg_h_r_pairs_dict[(h, r)]
        loss = 0
        for (r1, r2) in pos_r_pairs:
            for (r3, r4) in neg_r_pairs:
                r1_embedding = self.relation_embeddings[r1]
                r2_embedding = self.relation_embeddings[r2]
                r3_embedding = self.relation_embeddings[r3]
                r4_embedding = self.relation_embeddings[r4]
                pos_distance_vec = np.dot(r1_embedding, self.matrix) - r2_embedding
                neg_distance_vec = np.dot(r3_embedding, self.matrix) - r4_embedding
                pos_distance = get_distance(pos_distance_vec, self.norm)
                neg_distance = get_distance(neg_distance_vec, self.norm)
                step_loss = self.k * max(self.margin + pos_distance - neg_distance, 0)
                loss += step_loss
                # 只在loss为正时更新向量表示
                if step_loss > 0:
                    pos_gradient = 2 * self.k * pos_distance_vec
                    neg_gradient = 2 * self.k * neg_distance_vec

                    # 如果是L1范式再改变梯度的具体值
                    if self.norm == "L1":
                        for i in range(len(pos_gradient)):
                            if pos_gradient[i] > 0:
                                pos_gradient[i] = self.k
                            else:
                                pos_gradient[i] = -self.k

                            if neg_gradient[i] > 0:
                                neg_gradient[i] = self.k
                            else:
                                neg_gradient[i] = -self.k

                    r1_embedding -= self.learning_rate * pos_gradient
                    r2_embedding += self.learning_rate * pos_gradient
                    r3_embedding += self.learning_rate * neg_gradient
                    r4_embedding -= self.learning_rate * neg_gradient
                    self.relation_embeddings[r1] = r1_embedding
                    self.relation_embeddings[r2] = r2_embedding
                    self.relation_embeddings[r3] = r3_embedding
                    self.relation_embeddings[r4] = r4_embedding

        return loss

    def _generate_relation_pairs(self):
        # 过滤出头实体相同且有时间标记的四元组
        h_quads_dict = defaultdict(list)
        for (h, r, t, d) in self.kg.train_quads:
            if float(d) > 0:
                h_quads_dict[h].append((h, r, t, float(d)))
        # 如果(h1,r1,t1,d1)和(h2,r2,t2,d2)中，h1=h2且r1!=r2且d1<d2，则将(r1,r2)加入结果集
        for quads in h_quads_dict.values():
            if len(quads) > 1:
                i = 0
                for (h1, r1, t1, d1) in quads:
                    j = 0
                    for (h2, r2, t2, d2) in quads:
                        if i < j and r1 != r2 and d1 < d2:
                            self.pos_h_r_pairs_dict[(h1, r1)].add((r1, r2))
                            self.neg_h_r_pairs_dict[(h1, r1)].add((r2, r1))
                        j += 1
                    i += 1
        # print(len(self.pos_h_r_pairs_dict))
        # print(self.pos_h_r_pairs_dict)
