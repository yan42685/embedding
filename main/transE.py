import os
import codecs
import numpy as np
import copy
import time
import random


def main(data_set_name):
    entity_file = ""
    relation_file = ""
    fact_file = ""
    if data_set_name == "free_base":
        entity_file = "data_set/FB15k/entity2id.txt"
        relation_file = "data_set/FB15k/relation2id.txt"
        fact_file = "data_set/FB15k/test.txt"
    elif data_set_name == "word_net":
        entity_file = "data_set/WN18/entity2id.txt"
        relation_file = "data_set/WN18/relation2id.txt"
        # 训练耗时：test集13秒，train集100秒
        fact_file = "data_set/WN18/wordnet-mlj12-test.txt"
    else:
        raise RuntimeError("Wrong data set name")
    entity_ids, relation_ids, facts = load_data(entity_file, relation_file, fact_file)

    model = TransE(entity_ids, relation_ids, facts, dimension=50, learning_rate=0.01, margin=1.0, norm=2)
    model.train(data_set_name=data_set_name)


def load_data(entity_file, relation_file, fact_file):
    print("loading files...")

    entity_id_dict = {}
    relation_id_dict = {}
    entity_ids = []
    relation_ids = []
    with open(entity_file, "r") as file1, open(relation_file, "r") as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()
        for line in lines1:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            entity_id_dict[line[0]] = line[1]
            entity_ids.append(line[1])

        for line in lines2:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            relation_id_dict[line[0]] = line[1]
            relation_ids.append(line[1])

    facts = []

    with codecs.open(fact_file, "r") as file3:
        lines3 = file3.readlines()
        for line in lines3:
            fact = line.strip().split("\t")
            if len(fact) != 3:
                continue

            head_id = entity_id_dict[fact[0]]
            relation_id = relation_id_dict[fact[1]]
            tail_id = entity_id_dict[fact[2]]

            facts.append([head_id, relation_id, tail_id])

    print("Loading complete. entity : %d , relation : %d , fact : %d" % (
        len(entity_ids), len(relation_ids), len(facts)))

    return entity_ids, relation_ids, facts


# 曼哈顿距离
def norm_l1(vector):
    return np.linalg.norm(vector, ord=1)


# 欧氏距离
def norm_l2(vector):
    return np.linalg.norm(vector, ord=2)


# 缩小到欧氏距离的单位长度
def scale_to_unit_length(vector):
    return vector / norm_l2(vector)


class TransE:
    def __init__(self, entities, relations, facts, dimension=50, learning_rate=0.01, margin=1.0, norm=1):
        self.entities = entities
        self.relations = relations
        self.entity_vector_dict = {}
        self.relation_vector_dict = {}
        self.facts = facts
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self._vector_init()

    def _vector_init(self):
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            self.entity_vector_dict[entity] = entity_vector

        for relation in self.relations:
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                                self.dimension)
            relation_vector = scale_to_unit_length(relation_vector)
            # 不知道为什么，换成下面这行代码反而lose降低了10%左右
            # relation_vector = norm_l2(relation_vector)
            self.relation_vector_dict[relation] = relation_vector

    def train(self, epoch_count=1, batch_count=100, data_set_name=""):

        batch_size = int(len(self.facts) / batch_count)
        print("batch size: ", batch_size)
        for epoch in range(epoch_count):
            start_time = time.time()
            self.loss = 0.0
            for entity in self.entity_vector_dict.keys():
                self.entity_vector_dict[entity] = scale_to_unit_length(self.entity_vector_dict[entity])

            for batch in range(batch_count):
                positive_samples = random.sample(self.facts, batch_size)

                sample_pairs = []
                # 根据正例构建负例
                for positive_sample in positive_samples:
                    negative_sample = copy.deepcopy(positive_sample)
                    # 随机替换正例头实体或尾实体, 得到对应的负例
                    random_choice = np.random.random(1)[0]
                    if random_choice > 0.5:
                        # 替换正例的头实体
                        negative_sample[0] = random.sample(self.entities, 1)[0]
                        while negative_sample[0] == positive_sample[0]:
                            negative_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        # 替换正例的尾实体
                        negative_sample[2] = random.sample(self.entities, 1)[0]
                        while negative_sample[2] == positive_sample[2]:
                            negative_sample[2] = random.sample(self.entities, 1)[0]

                    if (positive_sample, negative_sample) not in sample_pairs:
                        sample_pairs.append((positive_sample, negative_sample))

                self._update_embedding(sample_pairs)
            end_time = time.time()
            print("epoch: ", epoch_count, "cost time: %s" % (round((end_time - start_time), 3)))
            print("running loss: ", self.loss)

        self._output_result(data_set_name, batch_size)

    def _update_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity_vector_dict 和
        # entity_vector_dict中所有的元素都不同
        copy_entity_vector_dict = copy.deepcopy(self.entity_vector_dict)
        copy_relation_vector_dict = copy.deepcopy(self.relation_vector_dict)

        for positive_sample, negative_sample in Tbatch:

            positive_head = self.entity_vector_dict[positive_sample[0]]
            positive_tail = self.entity_vector_dict[positive_sample[2]]
            relation = self.relation_vector_dict[positive_sample[1]]

            negative_head = self.entity_vector_dict[negative_sample[0]]
            negative_tail = self.entity_vector_dict[negative_sample[2]]

            # 计算向量之间的距离
            if self.norm == 1:
                positive_distance = norm_l1(positive_head + relation - positive_tail)
                negative_distance = norm_l1(negative_head + relation - negative_tail)

            else:
                positive_distance = norm_l2(positive_head + relation - positive_tail)
                negative_distance = norm_l2(negative_head + relation - negative_tail)

            loss = self.margin + positive_distance - negative_distance
            if loss > 0:
                self.loss += loss

                # 默认是L2范式
                positive_gradient = 2 * (positive_head + relation - positive_tail)
                negative_gradient = 2 * (negative_head + relation - negative_tail)

                # 如果是L1范式再改变梯度的具体值
                if self.norm == 1:
                    for i in range(len(positive_gradient)):
                        if positive_gradient[i] > 0:
                            positive_gradient[i] = 1
                        else:
                            positive_gradient[i] = -1

                        if negative_gradient[i] > 0:
                            negative_gradient[i] = 1
                        else:
                            negative_gradient[i] = -1

                positive_head -= self.learning_rate * positive_gradient
                relation -= self.learning_rate * positive_gradient
                positive_tail -= -1 * self.learning_rate * positive_gradient

                relation -= -1 * self.learning_rate * negative_gradient
                # 如果负例替换的是尾实体，则将头实体的向量表示更新两次
                if positive_sample[0] == negative_sample[0]:
                    positive_head -= -1 * self.learning_rate * negative_gradient
                    negative_tail -= self.learning_rate * negative_gradient
                # 如果负例替换的是头实体，则将尾实体的向量表示更新两次
                elif positive_sample[2] == negative_sample[2]:
                    negative_head -= -1 * self.learning_rate * negative_gradient
                    positive_tail -= self.learning_rate * negative_gradient

                # 将头尾实体新的向量表示放缩到单位长度
                copy_entity_vector_dict[positive_sample[0]] = scale_to_unit_length(positive_head)
                copy_entity_vector_dict[positive_sample[2]] = scale_to_unit_length(positive_tail)
                if positive_sample[0] == negative_sample[0]:
                    # 如果负例替换的是尾实体，则更新尾实体的向量表示
                    copy_entity_vector_dict[negative_sample[2]] = scale_to_unit_length(negative_tail)
                elif positive_sample[2] == negative_sample[2]:
                    # 如果负例替换的是头实体，则更新头实体的向量表示
                    copy_entity_vector_dict[negative_sample[0]] = scale_to_unit_length(negative_head)
                # TransE论题提到关系的向量表示不用缩放到单位长度
                copy_relation_vector_dict[positive_sample[1]] = relation
                # copy_relation[correct_sample[1]] = self.normalization(relation_copy)

        self.entity_vector_dict = copy_entity_vector_dict
        self.relation_vector_dict = copy_relation_vector_dict

    def _output_result(self, data_set_name, batch_size):
        data_set_name = data_set_name + "_"
        target_dir = "target/"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        with codecs.open(
                target_dir + data_set_name + "TransE_entity_" + str(self.dimension) + "dim_batch" + str(batch_size),
                "w") as file1:

            for e in self.entity_vector_dict.keys():
                file1.write(e + "\t")
                file1.write(str(list(self.entity_vector_dict[e])))
                file1.write("\n")

        with codecs.open(
                target_dir + data_set_name + "TransE_relation_" + str(self.dimension) + "dim_batch" + str(batch_size),
                "w") as file2:
            for r in self.relation_vector_dict.keys():
                file2.write(r + "\t")
                file2.write(str(list(self.relation_vector_dict[r])))
                file2.write("\n")


if __name__ == "__main__":
    main("word_net")
