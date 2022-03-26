from tools import generate_initial_vector, norm_l1, norm_l2, scale_to_unit_length
import codecs
import os
import numpy as np
from abc import ABCMeta, abstractmethod
import time
import random


class BaseModel(metaclass=ABCMeta):
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
        self.total_loss = 0.0
        # 总的样本数，用于计算平均loss
        self.total_sample_count = 0

    def train(self, epoch_count=1, batch_size=50, data_set_name=""):
        self._init_embeddings()
        batch_count = int(len(self.facts) / batch_size)
        print("batch size: ", batch_size)
        for epoch in range(epoch_count):
            start_time = time.time()
            self.total_loss = 0.0
            self.total_sample_count = 0
            for entity in self.entity_vector_dict.keys():
                self.entity_vector_dict[entity] = scale_to_unit_length(self.entity_vector_dict[entity])

            for batch in range(batch_count):
                positive_batch, negative_batch = self._generate_pos_neg_batch(batch_size)
                self._update_embeddings(positive_batch, negative_batch)
            # 让学习率衰减
            self.learning_rate = pow(0.95, epoch + 1) * self.learning_rate
            end_time = time.time()
            print("epoch: ", epoch + 1, "  cost time: %.3fs" % (end_time - start_time))
            print("total loss: %.6f" % self.total_loss)
            print("average loss: %.6f" % (self.total_loss / self.total_sample_count))
            print()

        # self._output_result(data_set_name, batch_size)

    def _init_embeddings(self):
        for entity in self.entities:
            self.entity_vector_dict[entity] = generate_initial_vector(self.dimension)

        for relation in self.relations:
            relation_vector = scale_to_unit_length(generate_initial_vector(self.dimension))
            self.relation_vector_dict[relation] = relation_vector

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

    @abstractmethod
    def _update_embeddings(self, positive_samples, negative_samples):
        pass

    # def _output_result(self, data_set_name, batch_size):
    #     data_set_name = data_set_name + "_"
    #     target_dir = "target/"
    #     if not os.path.exists(target_dir):
    #         os.mkdir(target_dir)
    #
    #     with codecs.open(
    #             target_dir + data_set_name + "transe_entity_" + str(self.dimension) + "dim_batch" + str(batch_size),
    #             "w") as file1:
    #
    #         for e in self.entity_vector_dict.keys():
    #             file1.write(e + "\t")
    #             file1.write(str(list(self.entity_vector_dict[e])))
    #             file1.write("\n")
    #
    #     with codecs.open(
    #             target_dir + data_set_name + "transe_relation_" + str(self.dimension) + "dim_batch" + str(batch_size),
    #             "w") as file2:
    #         for r in self.relation_vector_dict.keys():
    #             file2.write(r + "\t")
    #             file2.write(str(list(self.relation_vector_dict[r])))
    #             file2.write("\n")
