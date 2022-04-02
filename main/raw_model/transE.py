from raw_model.base_model import BaseModel
import copy
from tools import scale_to_unit_length, get_distance


class TransE(BaseModel):
    def __init__(self, kg_dir, model_name="TransE", epochs=1, batch_size=50, dimension=50, learning_rate=0.01,
                 margin=1.0,
                 norm="L1", epsilon=0.9, evaluation_mode="validation"):
        super().__init__(kg_dir, model_name, epochs, batch_size, dimension, learning_rate, margin, norm, epsilon,
                         evaluation_mode)

    def _update_embeddings(self, positive_samples, negative_samples):
        self.total_sample_count += len(positive_samples) * len(negative_samples)

        for positive_sample in positive_samples:
            for negative_sample in negative_samples:

                pos_h = self.entity_embeddings[positive_sample[0]]
                pos_t = self.entity_embeddings[positive_sample[2]]
                r = self.relation_embeddings[positive_sample[1]]

                neg_h = self.entity_embeddings[negative_sample[0]]
                neg_t = self.entity_embeddings[negative_sample[2]]

                # 计算向量之间的距离
                pos_distance = get_distance(pos_h + r - pos_t, self.norm)
                neg_distance = get_distance(neg_h + r - neg_t, self.norm)

                loss = self.margin + pos_distance - neg_distance
                if loss > 0:
                    self.total_loss += loss

                    # 默认是L2范式
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
