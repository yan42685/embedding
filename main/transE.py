import os
import codecs
import numpy as np
import copy
import time
import random


def main(data_set_name):
    if data_set_name == "free_base":
        entity_file = "data_set/FB15k/entity2id.txt"
        relation_file = "data_set/FB15k/relation2id.txt"
        fact_file = "data_set/FB15k/test.txt"
    elif data_set_name == "word_net":
        entity_file = "data_set/WN18/entity2id.txt"
        relation_file = "data_set/WN18/relation2id.txt"
        fact_file = "data_set/WN18/wordnet-mlj12-test.txt"
    else:
        raise RuntimeError("Wrong data set name")
    entity_ids, relation_ids, facts = load_data(entity_file, relation_file, fact_file)

    model = TransE(entity_ids, relation_ids, facts, dimension=50, learning_rate=0.01, margin=1.0, norm=2)
    model.train(epoch_count=1, data_set_name=data_set_name)


def load_data(entity_file, relation_file, fact_file):
    print("loading files...")

    entities = []
    relations = []
    facts = []

    with codecs.open(entity_file, "r") as file1, codecs.open(relation_file, "r") as file2, codecs.open(fact_file,
                                                                                                       "r") as file3:
        lines1 = file1.readlines()
        for line in lines1:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            entities.append(line[0])

        lines2 = file2.readlines()
        for line in lines2:
            line = line.strip().split("\t")
            if len(line) != 2:
                continue
            relations.append(line[0])

        lines3 = file3.readlines()
        for line in lines3:
            fact = line.strip().split("\t")
            if len(fact) != 3:
                continue
            facts.append(fact)

    print("Loading complete. entity : %d , relation : %d , fact : %d" % (
        len(entities), len(relations), len(facts)))

    return entities, relations, facts


# 曼哈顿距离
def norm_l1(vector):
    return np.linalg.norm(vector, ord=1)


# 欧氏距离
def norm_l2(vector):
    return np.linalg.norm(vector, ord=2)


# 缩放到欧氏距离的单位长度
def scale_to_unit_length(vector):
    return vector / norm_l2(vector)


class TransE:
    def __init__(self, entities, relations, facts, dimension=50, learning_rate=0.01, margin=1.0, norm=1):
        self.entities = entities
        self.entities_set = set(entities)
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
                positive_batch, negative_batch = self._generate_pos_neg_batch(batch_size)
                self._update_embedding(positive_batch, negative_batch)
            # 让学习率衰减
            self.learning_rate = pow(0.95, epoch + 1) * self.learning_rate
            end_time = time.time()
            print("epoch: ", epoch + 1, "cost time: %s" % (round((end_time - start_time), 3)))
            print("total loss: ", self.loss)

        self._output_result(data_set_name, batch_size)

    def _generate_pos_neg_batch(self, batch_size):
        positive_batch = random.sample(self.facts, batch_size)
        negative_batch = []

        # 随机替换正例头实体或尾实体, 得到对应的一个负例
        for (head, relation, tail) in positive_batch:
            random_choice = np.random.random()
            while True:
                if random_choice <= 0.5:
                    head = random.choice(self.entities)
                else:
                    tail = random.choice(self.entities)
                # 确保负例不在原facts中
                if (head, relation, tail) not in self.entities_set:
                    break
            negative_batch.append((head, relation, tail))

        return positive_batch, negative_batch

    def _update_embedding(self, positive_samples, negative_samples):
        copy_entity_vector_dict = copy.copy(self.entity_vector_dict)
        copy_relation_vector_dict = copy.copy(self.relation_vector_dict)

        for positive_sample in positive_samples:
            for negative_sample in negative_samples:

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
                    copy_entity_vector_dict[positive_sample[0]] = scale_to_unit_length(positive_head)
                    copy_entity_vector_dict[positive_sample[2]] = scale_to_unit_length(positive_tail)
                    # 将负例中被替换的头实体或尾实体的新向量表示放缩到单位长度, 并替换原来的向量表示
                    if positive_sample[0] != negative_sample[0]:
                        copy_entity_vector_dict[negative_sample[0]] = scale_to_unit_length(negative_head)
                    elif positive_sample[2] != negative_sample[2]:
                        copy_entity_vector_dict[negative_sample[2]] = scale_to_unit_length(negative_tail)

                    # TransE论文提到关系的向量表示不用缩放到单位长度
                    copy_relation_vector_dict[positive_sample[1]] = relation

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
