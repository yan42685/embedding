from raw_model.base_model import BaseModel
from tools import generate_initial_vector
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
            self.relation_normal_embeddings.append(generate_initial_vector(self.dimension))
            self.relation_hyper_embeddings.append(generate_initial_vector(self.dimension))

    def _update_embeddings(self, positive_samples, negative_samples):
        pass

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


# 其他实现说正交约束影响很小, 所以只报保留模长约束，舍弃正交约束
def _unit_constraint(pos_h, pos_t, neg_h, neg_t):
    return np.linalg.norm(pos_h) ** 2 - 1 + np.linalg.norm(pos_t) ** 2 - 1 + np.linalg.norm(
        neg_h) ** 2 - 1 + np.linalg.norm(
        neg_t) ** 2 - 1
