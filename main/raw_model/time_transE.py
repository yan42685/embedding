from raw_model.base_model import BaseModel
from collections import defaultdict
from tools import norm_l1, norm_l2, scale_to_unit_length


class TimeTransE(BaseModel):
    def __init__(self, kg_dir):
        super().__init__(kg_dir)
        # 每个头实体对应的(ri, rj)时序关系对集合
        self.pos_h_r_pairs_dict = defaultdict(set)
        self.neg_h_r_pairs_dict = defaultdict(set)
        self._generate_relation_pairs()

    def _update_embeddings(self, positive_samples, negative_samples):
        self.total_sample_count += len(positive_samples) * len(negative_samples)

        for positive_sample in positive_samples:
            for negative_sample in negative_samples:

                positive_head = self.entity_embeddings[positive_sample[0]]
                positive_tail = self.entity_embeddings[positive_sample[2]]
                relation = self.relation_embeddings[positive_sample[1]]

                negative_head = self.entity_embeddings[negative_sample[0]]
                negative_tail = self.entity_embeddings[negative_sample[2]]

                # 计算向量之间的距离
                if self.norm == "L1":
                    positive_distance = norm_l1(positive_head + relation - positive_tail)
                    negative_distance = norm_l1(negative_head + relation - negative_tail)

                else:
                    positive_distance = norm_l2(positive_head + relation - positive_tail)
                    negative_distance = norm_l2(negative_head + relation - negative_tail)

                loss = self.margin + positive_distance - negative_distance
                if loss > 0:
                    self.total_loss += loss

                    # 默认是L2范式
                    positive_gradient = 2 * (positive_head + relation - positive_tail)
                    negative_gradient = 2 * (negative_head + relation - negative_tail)

                    # 如果是L1范式再改变梯度的具体值
                    if self.norm == "L1":
                        for i in range(len(positive_gradient)):
                            if positive_gradient[i] > 0:
                                positive_gradient[i] = 1
                            else:
                                positive_gradient[i] = -1

                            if negative_gradient[i] > 0:
                                negative_gradient[i] = 1
                            else:
                                negative_gradient[i] = -1

                    # 损失函数希望达到的理想情况是，正例的d(h + r, t) 尽可能小，负例的d(h' + r', t') 尽可能大，
                    # 这样才能让总体的loss趋向于0。因此在梯度下降过程中，
                    # 正例中h和r逐渐减小，但t逐渐增大； 负例中h'和r'逐渐增大，而t'逐渐减小
                    positive_head -= self.learning_rate * positive_gradient
                    positive_tail += self.learning_rate * positive_gradient

                    # 如果负例替换的是头实体, 则正例的尾实体更新两次, 一次满足正例，一次满足负例
                    if positive_sample[0] != negative_sample[0]:
                        negative_head += self.learning_rate * negative_gradient
                        positive_tail -= self.learning_rate * negative_gradient
                    # 如果负例替换的是尾实体, 则正例的头实体更新两次
                    elif positive_sample[2] != negative_sample[2]:
                        negative_tail -= self.learning_rate * negative_gradient
                        positive_head += self.learning_rate * negative_gradient

                    # 关系永远更新两次
                    relation -= self.learning_rate * positive_gradient
                    relation += self.learning_rate * negative_gradient

                    # 将正例头尾实体新的向量表示放缩到单位长度, 并替换原来的向量表示
                    self.entity_embeddings[positive_sample[0]] = scale_to_unit_length(positive_head)
                    self.entity_embeddings[positive_sample[2]] = scale_to_unit_length(positive_tail)
                    # 将负例中被替换的头实体或尾实体的新向量表示放缩到单位长度, 并替换原来的向量表示
                    if positive_sample[0] != negative_sample[0]:
                        self.entity_embeddings[negative_sample[0]] = scale_to_unit_length(negative_head)
                    elif positive_sample[2] != negative_sample[2]:
                        self.entity_embeddings[negative_sample[2]] = scale_to_unit_length(negative_tail)

                    # TransE论文提到关系的向量表示不用缩放到单位长度
                    self.relation_embeddings[positive_sample[1]] = relation

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
                            self.pos_h_r_pairs_dict[h1].add((r1, r2))
                            self.neg_h_r_pairs_dict[h1].add((r2, r1))
                        j += 1
                    i += 1
        # print(len(self.pos_h_r_pairs_dict))
        # print(self.pos_h_r_pairs_dict)
