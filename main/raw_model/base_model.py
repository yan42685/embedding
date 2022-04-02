from tools import generate_initial_vector, scale_to_unit_length, time_it
import numpy as np
from abc import ABCMeta, abstractmethod
from raw_model.evaluator import Evaluator
from raw_model.KG import KG
import time
import random


class BaseModel(metaclass=ABCMeta):
    def __init__(self, kg_dir, model_name, epochs=1, batch_size=50, dimension=50, learning_rate=0.01, margin=1.0,
                 norm="L1",
                 epsilon=0.9,
                 evaluation_mode="validation"):
        self.kg = KG(directory=kg_dir)
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.dimension = dimension
        # 用于后续打印超参数，只记录learning_rate的话，最后打印的不是初始值
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.margin = margin
        self.norm = norm
        self.epsilon = epsilon
        self.evaluation_mode = evaluation_mode

        self.entity_embeddings = []
        self.relation_embeddings = []
        self.total_loss = 0.0
        # 总的样本数，用于计算平均loss
        self.total_sample_count = 0

    @time_it
    def train(self):
        self._init_embeddings()
        batch_count = int(len(self.kg.train_quads) / self.batch_size)
        for epoch in range(self.epochs):
            start_time = time.time()
            self.total_loss = 0.0
            self.total_sample_count = 0
            for i in range(len(self.entity_embeddings)):
                self.entity_embeddings[i] = scale_to_unit_length(self.entity_embeddings[i])

            for batch in range(batch_count):
                positive_batch, negative_batch = self._generate_pos_neg_batch(self.batch_size)
                self._update_embeddings(positive_batch, negative_batch)
            # 让学习率衰减
            self.learning_rate = pow(0.95, epoch + 1) * self.learning_rate
            end_time = time.time()
            print("epoch: ", epoch + 1, "  cost time: %.3fs" % (end_time - start_time))
            print("total loss: %.6f" % self.total_loss)
            if not self.total_sample_count == 0:
                print("average loss: %.6f" % (self.total_loss / self.total_sample_count))
            print()

        self._evaluate()

    def _init_embeddings(self):
        for _ in range(len(self.kg.entity_ids)):
            self.entity_embeddings.append(generate_initial_vector(self.dimension))

        for _ in range(len(self.kg.relation_ids)):
            self.relation_embeddings.append(scale_to_unit_length(generate_initial_vector(self.dimension)))

    def _generate_pos_neg_batch(self, batch_size):
        positive_batch = random.sample(self.kg.train_quads, batch_size)
        negative_batch = []

        # 随机替换正例头实体或尾实体, 得到对应的一个负例
        for (head, relation, tail, date) in positive_batch:
            random_choice = np.random.random()
            while True:
                if random_choice <= 0.5:
                    head = random.choice(self.kg.entity_ids)
                else:
                    tail = random.choice(self.kg.entity_ids)
                # 确保负例不在原facts中
                if (head, relation, tail) not in self.kg.train_quads_set:
                    break
            negative_batch.append((head, relation, tail, date))

        return positive_batch, negative_batch

    @abstractmethod
    def _update_embeddings(self, positive_samples, negative_samples):
        pass

    def _evaluate(self):
        if self.evaluation_mode == "validation":
            Evaluator(self.entity_embeddings, self.relation_embeddings, self.kg.train_quads,
                      self.kg.validation_quads, self.norm, self.epsilon).evaluate()
        elif self.evaluation_mode == "test":
            Evaluator(self.entity_embeddings, self.relation_embeddings, self.kg.train_quads,
                      self.kg.test_quads, self.norm, self.epsilon).evaluate()
        else:
            raise RuntimeError("wrong evaluation_mode")

        print(
            "%s epochs=%d, batch_size=%d, dimension=%d, learning_rate=%.4f, margin=%.1f, norm=%s, " % (self.model_name,
                                                                                                       self.epochs,
                                                                                                       self.batch_size,
                                                                                                       self.dimension,
                                                                                                       self.initial_learning_rate,
                                                                                                       self.margin,
                                                                                                       self.norm,))
        print("epsilon=%.2f, evaluation_mode=%s" % (self.epsilon, self.evaluation_mode))
