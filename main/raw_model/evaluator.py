from tools import norm_l1, norm_l2, time_it
import time
import scipy.stats as ss


# 评估模型训练的效果

class Evaluator:
    def __init__(self, entity_embeddings, relation_embeddings, test_quads, all_quads, norm="L1"):
        self.entity_embeddings = entity_embeddings
        self.entity_count = len(self.entity_embeddings)
        self.relation_embeddings = relation_embeddings
        self.test_quads = test_quads
        self.all_quads_set = set(all_quads)
        self.norm = norm

    def evaluate(self):
        print("Start evaluating...")
        self._calculate_mean_rank_and_hits10()

    # 大概三分钟
    @time_it
    def _calculate_mean_rank_and_hits10(self):
        raw_h_correct_rank_sum = 0
        raw_h_hits10_count = 0
        raw_t_correct_rank_sum = 0
        raw_t_hits10_count = 0
        filter_h_correct_rank_sum = 0
        filter_h_hits10_count = 0
        filter_t_correct_rank_sum = 0
        filter_t_hits10_count = 0
        for quad in self.test_quads:
            a, b, c, d, e, f, g, h = self._calculate_rank_data(quad)
            print("result: %d, %d, %d, %d, %d, %d, %d, %d" % (a, b, c, d, e, f, g, h))
            raw_h_correct_rank_sum += a
            raw_h_hits10_count += b
            raw_t_correct_rank_sum += c
            raw_t_hits10_count += d
            filter_h_correct_rank_sum += e
            filter_h_hits10_count += f
            filter_t_correct_rank_sum += g
            filter_t_hits10_count += h

        all_quads_count = len(self.all_quads_set)
        print("-----Raw-----")
        print("-----head prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (raw_h_correct_rank_sum / all_quads_count,
                                                   100 * raw_h_hits10_count / all_quads_count))
        print("-----tail prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (raw_t_correct_rank_sum / all_quads_count,
                                                   100 * raw_t_hits10_count / all_quads_count))
        print("------Raw Average------")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % ((raw_h_correct_rank_sum / all_quads_count +
                                                    raw_t_correct_rank_sum / all_quads_count) / 2,
                                                   100 * (raw_h_hits10_count / all_quads_count +
                                                          raw_t_hits10_count / all_quads_count) / 2))

        print()
        print("-----Filter-----")
        print("-----head prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (filter_h_correct_rank_sum / all_quads_count,
                                                   100 * filter_h_hits10_count / all_quads_count))
        print("-----tail prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (filter_t_correct_rank_sum / all_quads_count,
                                                   100 * filter_t_hits10_count / all_quads_count))
        print("------Filter Average------")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % ((filter_h_correct_rank_sum / all_quads_count +
                                                    filter_t_correct_rank_sum / all_quads_count) / 2,
                                                   100 * (filter_h_hits10_count / all_quads_count +
                                                          filter_t_hits10_count / all_quads_count) / 2))

    # 按预测距离排序实体id
    def _calculate_sorted_ids_by_prediction_distance(self, test_quad):
        h = self.entity_embeddings[test_quad[0]]
        r = self.relation_embeddings[test_quad[1]]
        t = self.entity_embeddings[test_quad[2]]
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
            else:
                raise RuntimeError("wrong norm")

        h_sorted_ids = sorted(list(range(self.entity_count)), key=lambda x: h_predict_distances[x])
        t_sorted_ids = sorted(list(range(self.entity_count)), key=lambda x: t_predict_distances[x])

        return h_sorted_ids, t_sorted_ids

    def _calculate_rank_data(self, test_quad):
        raw_h_correct_rank = 1
        raw_h_hits10 = 0
        raw_t_correct_rank = 1
        raw_t_hits10 = 0
        filter_h_correct_rank = 1
        filter_h_hits10 = 0
        filter_t_correct_rank = 1
        filter_t_hits10 = 0

        (h, r, t, d) = test_quad
        h_sorted_ids, t_sorted_ids = self._calculate_sorted_ids_by_prediction_distance(test_quad)
        for h_predict_id in h_sorted_ids:
            if h_predict_id == h:
                break
            else:
                raw_h_correct_rank += 1
                if not (h_predict_id, r, t, d) in self.all_quads_set:
                    filter_h_correct_rank += 1

        for t_predict_id in t_sorted_ids:
            if t_predict_id == t:
                break
            else:
                raw_t_correct_rank += 1
                if not (h, r, t_predict_id, d) in self.all_quads_set:
                    filter_t_correct_rank += 1

        if raw_h_correct_rank <= 10:
            raw_h_hits10 = 1
        if raw_t_correct_rank <= 10:
            raw_t_hits10 = 1
        if filter_h_correct_rank <= 10:
            filter_h_hits10 = 1
        if filter_t_correct_rank <= 10:
            filter_t_hits10 = 1

        return raw_h_correct_rank, raw_h_hits10, raw_t_correct_rank, raw_t_hits10, \
               filter_h_correct_rank, filter_h_hits10, filter_t_correct_rank, filter_t_hits10
