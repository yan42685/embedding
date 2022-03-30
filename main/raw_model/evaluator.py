from tools import norm_l1, norm_l2


# 评估模型训练的效果

class Evaluator:
    def __init__(self, entity_embeddings, relation_embeddings, test_quads, norm="L1"):
        self.entity_embeddings = entity_embeddings
        self.entity_count = len(self.entity_embeddings)
        self.relation_embeddings = relation_embeddings
        self.test_quads = test_quads
        self.norm = norm

    def evaluate(self):
        pass

    # 对于某一个测试quad，计算每个实体在头实体预测和尾实体预测中的排名
    def _get_h_t_predict_ranks(self, quad):
        h = self.entity_embeddings[quad[0]]
        r = self.relation_embeddings[quad[1]]
        t = self.entity_embeddings[quad[2]]
        h_predict_distances = []
        t_predict_distances = []
        for i in range(self.entity_count):
            h_predict_distance = self.entity_embeddings[i] + r - t
            t_predict_distance = h + r - self.entity_embeddings[i]
            if self.norm == "L1":
                h_predict_distances.append(norm_l1(h_predict_distance))
                t_predict_distances.append(norm_l1(t_predict_distance))
            elif self.norm == "L2":
                h_predict_distances.append(norm_l2(h_predict_distance))
                t_predict_distances.append(norm_l2(t_predict_distance))

        # 得到每个实体预测距离的排名，排名从1开始，比如[5,2,4,1] -> [4,2,3,1]
        h_predict_ranks = [sorted(h_predict_distances).index(x) + 1 for x in h_predict_distances]
        t_predict_ranks = [sorted(t_predict_distances).index(x) + 1 for x in t_predict_distances]

        return h_predict_ranks, t_predict_ranks
