from tools import norm_l1, norm_l2, time_it


# 评估模型训练的效果

class Evaluator:
    def __init__(self, entity_embeddings, relation_embeddings, test_quads, norm="L1"):
        self.entity_embeddings = entity_embeddings
        self.entity_count = len(self.entity_embeddings)
        self.relation_embeddings = relation_embeddings
        self.test_quads = test_quads
        self.norm = norm

    def evaluate(self):
        self._calculate_mean_rank_and_hits10()

    def _calculate_mean_rank_and_hits10(self):
        h_correct_rank_sum = 0
        t_correct_rank_sum = 0
        h_hits10_count = 0
        t_hits10_count = 0
        for quad in self.test_quads:
            h_predict_ranks, t_predict_ranks = self._calculate_h_t_predict_ranks(quad)
            h_correct_rank_sum += h_predict_ranks[quad[0]]
            t_correct_rank_sum += t_predict_ranks[quad[2]]
            if h_predict_ranks[quad[0]] <= 10:
                h_hits10_count += 1
            if t_predict_ranks[quad[2]] <= 10:
                t_hits10_count += 1

        raw_mean_rank = (h_correct_rank_sum + t_correct_rank_sum) / (2 * len(self.test_quads))
        raw_hits10 = 100 * (h_hits10_count + t_hits10_count) / (2 * self.entity_count)

        print("raw mean rank: %.4f, raw hits10: %.4f%%" % (raw_mean_rank, raw_hits10))

    @time_it
    # 对于某一个测试quad，计算每个实体在头实体预测和尾实体预测中的排名
    def _calculate_h_t_predict_ranks(self, quad):
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
