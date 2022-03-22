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
    model.train(out_file_title=data_set_name + "_")


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
        self.vector_init()

    def vector_init(self):
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


    def train(self, epoch=1, batch=100, out_file_title=""):

        batch_size = int(len(self.facts) / batch)
        print("batch size: ", batch_size)
        for epoch in range(epoch):
            start_time = time.time()
            self.loss = 0.0
            for entity in self.entity_vector_dict.keys():
                self.entity_vector_dict[entity] = scale_to_unit_length(self.entity_vector_dict[entity]);

            for batch in range(batch):
                batch_samples = random.sample(self.facts, batch_size)

                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # change the head entity
                        corrupted_sample[0] = random.sample(self.entity_vector_dict.keys(), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entity_vector_dict.keys(), 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(self.entity_vector_dict.keys(), 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entity_vector_dict.keys(), 1)[0]

                    if (sample, corrupted_sample) not in Tbatch:
                        Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end_time = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end_time - start_time), 3)))
            print("running loss: ", self.loss)

        target_dir = "target/"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        with codecs.open(
                target_dir + out_file_title + "TransE_entity_" + str(self.dimension) + "dim_batch" + str(batch_size),
                "w") as f1:

            for e in self.entity_vector_dict.keys():
                f1.write(e + "\t")
                f1.write(str(list(self.entity_vector_dict[e])))
                f1.write("\n")

        with codecs.open(
                target_dir + out_file_title + "TransE_relation_" + str(self.dimension) + "dim_batch" + str(batch_size),
                "w") as f2:
            for r in self.relation_vector_dict.keys():
                f2.write(r + "\t")
                f2.write(str(list(self.relation_vector_dict[r])))
                f2.write("\n")

    def update_triple_embedding(self, Tbatch):
        # deepcopy 可以保证，即使list嵌套list也能让各层的地址不同， 即这里copy_entity 和
        # entitles中所有的elements都不同
        copy_entity = copy.deepcopy(self.entity_vector_dict)
        copy_relation = copy.deepcopy(self.relation_vector_dict)

        for correct_sample, corrupted_sample in Tbatch:

            correct_copy_head = copy_entity[correct_sample[0]]
            correct_copy_tail = copy_entity[correct_sample[2]]
            relation_copy = copy_relation[correct_sample[1]]

            corrupted_copy_head = copy_entity[corrupted_sample[0]]
            corrupted_copy_tail = copy_entity[corrupted_sample[2]]

            correct_head = self.entity_vector_dict[correct_sample[0]]
            correct_tail = self.entity_vector_dict[correct_sample[2]]
            relation = self.relation_vector_dict[correct_sample[1]]

            corrupted_head = self.entity_vector_dict[corrupted_sample[0]]
            corrupted_tail = self.entity_vector_dict[corrupted_sample[2]]

            # calculate the distance of the triples
            if self.norm == 1:
                correct_distance = norm_l1(correct_head + relation - correct_tail)
                corrupted_distance = norm_l1(corrupted_head + relation - corrupted_tail)

            else:
                correct_distance = norm_l2(correct_head + relation - correct_tail)
                corrupted_distance = norm_l2(corrupted_head + relation - corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            if loss > 0:
                self.loss += loss

                correct_gradient = 2 * (correct_head + relation - correct_tail)
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)

                if self.norm == 1:
                    for i in range(len(correct_gradient)):
                        if correct_gradient[i] > 0:
                            correct_gradient[i] = 1
                        else:
                            correct_gradient[i] = -1

                        if corrupted_gradient[i] > 0:
                            corrupted_gradient[i] = 1
                        else:
                            corrupted_gradient[i] = -1

                correct_copy_head -= self.learning_rate * correct_gradient
                relation_copy -= self.learning_rate * correct_gradient
                correct_copy_tail -= -1 * self.learning_rate * correct_gradient

                relation_copy -= -1 * self.learning_rate * corrupted_gradient
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replaces the tail entity, the head entity"s embedding need to be updated twice
                    correct_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    corrupted_copy_tail -= self.learning_rate * corrupted_gradient
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replaces the head entity, the tail entity"s embedding need to be updated twice
                    corrupted_copy_head -= -1 * self.learning_rate * corrupted_gradient
                    correct_copy_tail -= self.learning_rate * corrupted_gradient

                # normalising these new embedding vector, instead of normalising all the embedding together
                copy_entity[correct_sample[0]] = scale_to_unit_length(correct_copy_head)
                copy_entity[correct_sample[2]] = scale_to_unit_length(correct_copy_tail)
                if correct_sample[0] == corrupted_sample[0]:
                    # if corrupted_triples replace the tail entity, update the tail entity"s embedding
                    copy_entity[corrupted_sample[2]] = scale_to_unit_length(corrupted_copy_tail)
                elif correct_sample[2] == corrupted_sample[2]:
                    # if corrupted_triples replace the head entity, update the head entity"s embedding
                    copy_entity[corrupted_sample[0]] = scale_to_unit_length(corrupted_copy_head)
                # the paper mention that the relation"s embedding don"t need to be normalised
                copy_relation[correct_sample[1]] = relation_copy
                # copy_relation[correct_sample[1]] = self.normalization(relation_copy)

        self.entity_vector_dict = copy_entity
        self.relation_vector_dict = copy_relation


if __name__ == "__main__":
    main("word_net")
