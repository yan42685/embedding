from model.KG import KG
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class TfModel(metaclass=ABCMeta):
    def __init__(self, train_data, valid_data, test_data, embedding_dim, epochs=1, batch_size=100, margin=4,
                 learning_rate=0.001,
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

    @abstractmethod
    def loss_function(self):
        pass

    def train(self):
        optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate)
        model = tf.keras.Model()
        model.fit(epochs=self.epochs, batch_size=self.batch_size)
        model.compile(optimizer=optimizer, loss=self.loss_function)
        return model.evaluate()
