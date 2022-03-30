from tf_model.KG import KG
from abc import ABCMeta, abstractmethod
from tools import time_it
import time
import tensorflow as tf
import numpy as np
import random
import math


class BaseModel(metaclass=ABCMeta):
    def __init__(self, kg=KG(), embedding_dim=50, epochs=3, batch_size=50,
                 margin=2,
                 learning_rate=0.01,
                 norm="L1"):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = tf.constant(margin, dtype=tf.float32)
        self.learning_rate = learning_rate
        self.norm = norm

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.entity_embeddings = []
        self.relation_embeddings = []
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
            # 只有entity需要在每个epoch进行normalization, 而relation不需要
            for i in range(len(self.entity_embeddings)):
                self.entity_embeddings[i] = tf.Variable(tf.math.l2_normalize(self.entity_embeddings[i]))

            for _ in range(batch_count):
                pos_batch, neg_batch = self._generate_pos_neg_batch(self.batch_size)
                self._update_embedding(pos_batch, neg_batch)
                print("total loss: %.6f, average loss: %.6f" % (
                    self.total_loss, self.total_loss / self.total_sample_count))

            end_time = time.time()
            print("epoch: ", epoch + 1, "  cost time: %.3fs" % (end_time - start_time))
            print("total loss: %.6f, average loss: %.6f" % (self.total_loss, self.total_loss / self.total_sample_count))
            print()

    def _init_embedding(self):
        bound = 6 / math.sqrt(self.embedding_dim)

        for _ in range(len(self.kg.entity_ids)):
            self.entity_embeddings.append(
                tf.Variable(tf.random.uniform(shape=[self.embedding_dim], minval=-bound, maxval=bound)))
        for _ in range(len(self.kg.relation_ids)):
            self.relation_embeddings.append(
                tf.Variable(
                    tf.math.l2_normalize(tf.random.uniform(shape=[self.embedding_dim], minval=-bound, maxval=bound))))

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

    @time_it
    def _update_embedding(self, pos_quads, neg_quads):
        for pos_quad in pos_quads:
            for neg_quad in neg_quads:
                pos_h, pos_r, pos_t = self._lookup_embeddings(pos_quad)
                neg_h, neg_r, neg_t = self._lookup_embeddings(neg_quad)
                variables = [pos_h, pos_r, pos_t, neg_h, neg_r, neg_t]

                with tf.GradientTape() as tape:
                    loss = self._loss_function(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
                grads = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
                self.total_sample_count += 1
                self.total_loss += loss.numpy()

    # @time_it
    def _lookup_embeddings(self, quad):
        head = self.entity_embeddings[quad[0]]
        relation = self.relation_embeddings[quad[1]]
        tail = self.entity_embeddings[quad[2]]
        return head, relation, tail

    @abstractmethod
    def _loss_function(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pass
