from model.KG import KG
from abc import ABCMeta, abstractmethod
import time
import tensorflow as tf
import numpy as np
import random
import math


class TfModel(metaclass=ABCMeta):
    def __init__(self, train_data=KG(), embedding_dim=50, epochs=3, batch_size=100,
                 margin=4,
                 learning_rate=0.01,
                 norm="L1"):
        self.kg = train_data
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = tf.constant(margin, dtype=tf.float32)
        self.learning_rate = learning_rate
        self.norm = norm

        self.optimizer = None
        self.entities_embedding = None
        self.relations_embedding = None
        self.total_loss = 0.0
        # 总的样本数，用于计算平均loss
        self.total_sample_count = 0

    def train(self):
        self._init_embedding()
        batch_count = int(len(self.kg.train_quads) / self.batch_size)

        for epoch in range(self.epochs):
            start_time = time.time()
            self.total_loss = 0.0
            self.total_sample_count = 0
            # 让学习率衰减
            self.learning_rate = pow(0.95, epoch) * self.learning_rate
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
            # 只有entity需要在每个epoch进行normalization, 而relation不需要
            self.entities_embedding = tf.math.l2_normalize(self.entities_embedding, axis=1)

            for _ in range(batch_count):
                pos_batch, neg_batch = self._generate_pos_neg_batch(self.batch_size)
                self._update_embedding(pos_batch, neg_batch)

            end_time = time.time()
            print("epoch: ", epoch + 1, "  cost time: %.3fs" % (end_time - start_time))
            print("total loss: %.6f, average loss: %.6f" % (self.total_loss, self.total_loss / self.total_sample_count))
            print()

        # positive_quads, negative_quads = self.generate_pos_neg_batch()
        # optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        # for _ in range(self.epochs):
        #     with tf.GradientTape() as tape:
        #         loss = self._loss_function(tf.constant(), tf.constant())
        # grads = tape.gradient(loss, [pos])

        # model = tf.keras.Model()
        # model.compile(optimizer=optimizer, loss=self.loss_function)
        # model.fit(x=self.positive_quads, y=self.negative_quads, epochs=self.epochs, batch_size=self.batch_size)
        # total_loss = model.evaluate()

        # print("total_loss: %.4f" % total_loss)
        # return total_loss

    def _init_embedding(self):
        bound = 6 / math.sqrt(self.embedding_dim)
        uniform_initializer = tf.random_uniform_initializer(minval=-bound, maxval=bound)
        self.entities_embedding = tf.Variable(uniform_initializer(shape=[len(self.kg.entity_ids), self.embedding_dim]))
        self.relations_embedding = tf.Variable(uniform_initializer(shape=[len(self.kg.entity_ids), self.embedding_dim]))

        self.relations_embedding = tf.math.l2_normalize(self.relations_embedding, axis=1)

    @abstractmethod
    def _update_embedding(self, pos_quads, neg_quads):
        pass

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
                if (head, relation, tail, date) not in self.kg.train_quads_set:
                    break
            negative_batch.append((head, relation, tail, date))
        return positive_batch, negative_batch
