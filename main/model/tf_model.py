from model.KG import KG
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class TfModel(metaclass=ABCMeta):
    def __init__(self, train_data: KG, valid_data: KG, test_data: KG, embedding_dim=50, epochs=3, batch_size=100,
                 margin=4,
                 learning_rate=0.01,
                 norm="L1"):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.margin = margin
        self.learning_rate = learning_rate
        self.norm = norm

        self.entities_embedding = None
        self.relations_embedding = None

    @abstractmethod
    def loss_function(self):
        pass

    def train(self):
        """
        :return: total loss
        """
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        model = tf.keras.Model()
        model.fit(epochs=self.epochs, batch_size=self.batch_size)
        model.compile(optimizer=optimizer, loss=self.loss_function)
        return model.evaluate()

    def init_embedding(self):
        bound = 6 / pow(self.embedding_dim, 0.5)
        self.entities_embedding = tf.Variable(shape=[len(self.train_data.entities), self.embedding_dim],
                                              initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))
        self.relations_embedding = tf.Variable(shape=[len(self.train_data.relations), self.embedding_dim],
                                               initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound))

        self.entities_embedding = tf.math.l2_normalize(self.entities_embedding, axis=1)
        self.relations_embedding = tf.math.l2_normalize(self.relations_embedding, axis=1)
