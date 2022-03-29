from model.KG import KG
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import random


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

        self.entities_embedding = None
        self.relations_embedding = None
        self.positive_quads = []
        self.negative_quads = []

    @abstractmethod
    def loss_function(self, pos_quad: tuple, neg_quad: tuple):
        pass

    def train(self):
        """
        :return: total loss
        """
        self.positive_quads, self.negative_quads = self.generate_pos_neg_quads()
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        model = tf.keras.Model()
        model.compile(optimizer=optimizer, loss=self.loss_function)
        model.fit(x=self.positive_quads, y=self.negative_quads, epochs=self.epochs, batch_size=self.batch_size)
        total_loss = model.evaluate()
        print("total_loss: %.4f" % total_loss)
        return total_loss

    def init_embedding(self):
        bound = 6 / tf.math.sqrt(self.embedding_dim)
        self.entities_embedding = tf.Variable(shape=[self.kg.entity_count, self.embedding_dim],
                                              initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))
        self.relations_embedding = tf.Variable(shape=[self.kg.relation_count, self.embedding_dim],
                                               initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))

        self.entities_embedding = tf.math.l2_normalize(self.entities_embedding, axis=1)
        self.relations_embedding = tf.math.l2_normalize(self.relations_embedding, axis=1)

    def generate_pos_neg_quads(self):
        entities = list(self.kg.entity_id_dict.values())
        positive_quads = self.kg.train_quads
        positive_quads_set = set(positive_quads)
        negative_quads = []

        # 随机替换正例头实体或尾实体, 得到对应的一个负例
        for (head, relation, tail, date) in positive_quads:
            random_choice = np.random.random()
            while True:
                if random_choice <= 0.5:
                    head = random.choice(entities)
                else:
                    tail = random.choice(entities)
                # 确保负例不在原facts中
                if (head, relation, tail, date) not in positive_quads_set:
                    break
            negative_quads.append((head, relation, tail, date))
        return positive_quads, negative_quads
