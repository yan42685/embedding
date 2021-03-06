from tools import get_distance, time_it
import random
import numpy as np


# 评估模型训练的效果

class Evaluator:
    def __init__(self, entity_embeddings, relation_embeddings, train_quads, test_quads, norm, epsilon):
        self.entity_embeddings = entity_embeddings
        self.entity_count = len(entity_embeddings)
        self.relation_embeddings = relation_embeddings
        self.relation_count = len(relation_embeddings)
        self.all_quads_set = set(train_quads) | set(test_quads)
        self.train_quads = train_quads
        self.test_quads = test_quads
        self.norm = norm
        self.epsilon = epsilon

    def evaluate(self):
        print("Start evaluating...")
        self._triple_classification()
        self._link_predication()

    # @time_it
    def _triple_classification(self):
        test_heads = set()
        test_tails = set()
        for (h, r, t, d) in self.all_quads_set:
            test_heads.add(h)
            test_tails.add(t)
        test_heads = list(test_heads)
        test_tails = list(test_tails)

        corrupted_quads = []
        # 随机替换正例头实体或尾实体, 得到对应的一个负例
        for (head, relation, tail, date) in self.test_quads:
            random_choice = np.random.random()
            if random_choice <= 0.5:
                # 只选取出现在那个位置的实体
                head = random.choice(test_heads)
            else:
                tail = random.choice(test_tails)
            corrupted_quads.append((head, relation, tail, date))

        train_quads_distances = []
        for (h, r, t, d) in self.train_quads:
            trained_distance = self.entity_embeddings[h] + self.relation_embeddings[r] - self.entity_embeddings[t]
            train_quads_distances.append(get_distance(trained_distance, self.norm))
        train_quads_distances.sort()
        # 决定是否为正确的三元组的距离界限
        threshold = train_quads_distances[int(self.epsilon * len(train_quads_distances))]
        correct_prediction_count = 0

        for (h, r, t, d) in corrupted_quads:
            distance = get_distance(
                self.entity_embeddings[h] + self.relation_embeddings[r] - self.entity_embeddings[t], self.norm)
            condition1 = distance <= threshold and (h, r, t, d) in self.all_quads_set
            condition2 = distance > threshold and (h, r, t, d) not in self.all_quads_set
            if condition1 or condition2:
                correct_prediction_count += 1
        print("Triple classification accuracy: %.2f%%" % (100 * correct_prediction_count / len(self.test_quads)))

    @time_it
    def _link_predication(self):
        raw_h_correct_rank_sum = 0
        raw_h_hits10_count = 0
        raw_r_correct_rank_sum = 0
        raw_r_hits1_count = 0
        raw_t_correct_rank_sum = 0
        raw_t_hits10_count = 0
        filter_h_correct_rank_sum = 0
        filter_h_hits10_count = 0
        filter_r_correct_rank_sum = 0
        filter_r_hits1_count = 0
        filter_t_correct_rank_sum = 0
        filter_t_hits10_count = 0
        for quad in self.test_quads:
            a, b, c, d, e, f, g, h, i, j, k, l = self._calculate_rank_data(quad)
            # print("result: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d" % (a, b, c, d, e, f, g, h, i, j, k, l))
            raw_h_correct_rank_sum += a
            raw_h_hits10_count += b
            raw_r_correct_rank_sum += c
            raw_r_hits1_count += d
            raw_t_correct_rank_sum += e
            raw_t_hits10_count += f
            filter_h_correct_rank_sum += g
            filter_h_hits10_count += h
            filter_r_correct_rank_sum += i
            filter_r_hits1_count += j
            filter_t_correct_rank_sum += k
            filter_t_hits10_count += l

        test_quads_count = len(self.test_quads)
        print("=========== Entity Prediction =========")
        print("-----Raw-----")
        print("-----head prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (raw_h_correct_rank_sum / test_quads_count,
                                                   100 * raw_h_hits10_count / test_quads_count))
        print("-----tail prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (raw_t_correct_rank_sum / test_quads_count,
                                                   100 * raw_t_hits10_count / test_quads_count))
        print("------Raw Average------")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % ((raw_h_correct_rank_sum / test_quads_count +
                                                    raw_t_correct_rank_sum / test_quads_count) / 2,
                                                   100 * (raw_h_hits10_count / test_quads_count +
                                                          raw_t_hits10_count / test_quads_count) / 2))

        print()
        print("-----Filter-----")
        print("-----head prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (filter_h_correct_rank_sum / test_quads_count,
                                                   100 * filter_h_hits10_count / test_quads_count))
        print("-----tail prediction-----")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % (filter_t_correct_rank_sum / test_quads_count,
                                                   100 * filter_t_hits10_count / test_quads_count))
        print("------Filter Average------")
        print("MeanRank: %.1f, Hits@10: %.3f%%" % ((filter_h_correct_rank_sum / test_quads_count +
                                                    filter_t_correct_rank_sum / test_quads_count) / 2,
                                                   100 * (filter_h_hits10_count / test_quads_count +
                                                          filter_t_hits10_count / test_quads_count) / 2))

        print("=========== Relation Prediction =========")
        print("-----Raw-----")
        print("MeanRank: %.2f, Hits@1: %.3f%%" % (raw_r_correct_rank_sum / test_quads_count,
                                                  100 * raw_r_hits1_count / test_quads_count))
        print("-----Filter-----")
        print("MeanRank: %.2f, Hits@1: %.3f%%" % (filter_r_correct_rank_sum / test_quads_count,
                                                  100 * filter_r_hits1_count / test_quads_count))

    def _calculate_rank_data(self, test_quad):
        raw_h_correct_rank = 1
        raw_h_hits10 = 0
        raw_r_correct_rank = 1
        raw_r_hits1 = 0
        raw_t_correct_rank = 1
        raw_t_hits10 = 0
        filter_h_correct_rank = 1
        filter_h_hits10 = 0
        filter_r_correct_rank = 1
        filter_r_hits1 = 0
        filter_t_correct_rank = 1
        filter_t_hits10 = 0

        (h, r, t, d) = test_quad
        h_sorted_ids, r_sorted_ids, t_sorted_ids = self._calculate_sorted_ids_by_prediction_distance(test_quad)
        for h_predict_id in h_sorted_ids:
            if h_predict_id == h:
                break
            else:
                raw_h_correct_rank += 1
                if not (h_predict_id, r, t, d) in self.all_quads_set:
                    filter_h_correct_rank += 1

        for r_predict_id in r_sorted_ids:
            if r_predict_id == r:
                break
            else:
                raw_r_correct_rank += 1
                if not (h, r_predict_id, t, d) in self.all_quads_set:
                    filter_r_correct_rank += 1

        for t_predict_id in t_sorted_ids:
            if t_predict_id == t:
                break
            else:
                raw_t_correct_rank += 1
                if not (h, r, t_predict_id, d) in self.all_quads_set:
                    filter_t_correct_rank += 1

        if raw_h_correct_rank <= 10:
            raw_h_hits10 = 1
        if raw_r_correct_rank == 1:
            raw_r_hits1 = 1
        if raw_t_correct_rank <= 10:
            raw_t_hits10 = 1
        if filter_h_correct_rank <= 10:
            filter_h_hits10 = 1
        if raw_r_correct_rank == 1:
            filter_r_hits1 = 1
        if filter_t_correct_rank <= 10:
            filter_t_hits10 = 1

        return raw_h_correct_rank, raw_h_hits10, raw_r_correct_rank, raw_r_hits1, raw_t_correct_rank, raw_t_hits10, \
               filter_h_correct_rank, filter_h_hits10, filter_r_correct_rank, filter_r_hits1, filter_t_correct_rank, filter_t_hits10

    # 按预测距离排序实体id
    def _calculate_sorted_ids_by_prediction_distance(self, test_quad):
        h = self.entity_embeddings[test_quad[0]]
        r = self.relation_embeddings[test_quad[1]]
        t = self.entity_embeddings[test_quad[2]]
        h_predict_distances = []
        r_predict_distances = []
        t_predict_distances = []

        for i in range(self.entity_count):
            h_predict_distance = self.entity_embeddings[i] + r - t
            t_predict_distance = h + r - self.entity_embeddings[i]
            h_predict_distances.append(get_distance(h_predict_distance, self.norm))
            t_predict_distances.append(get_distance(t_predict_distance, self.norm))
        for i in range(self.relation_count):
            r_predict_distance = h + self.relation_embeddings[i] - t
            r_predict_distances.append(get_distance(r_predict_distance, self.norm))

        h_sorted_ids = sorted(list(range(self.entity_count)), key=lambda x: h_predict_distances[x])
        r_sorted_ids = sorted(list(range(self.relation_count)), key=lambda x: r_predict_distances[x])
        t_sorted_ids = sorted(list(range(self.entity_count)), key=lambda x: t_predict_distances[x])

        return h_sorted_ids, r_sorted_ids, t_sorted_ids
