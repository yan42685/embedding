from tf_model.base_model import BaseModel
import tensorflow as tf
from tools import time_it


class TransE(BaseModel):
    # @time_it
    def _loss_function(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_distance = pos_h + pos_r - pos_t
        neg_distance = neg_h + neg_r - neg_t
        if self.norm == "L1":
            pos_score = tf.reduce_sum(tf.abs(pos_distance))
            neg_score = tf.reduce_sum(tf.abs(neg_distance))
        elif self.norm == "L2":
            pos_score = tf.sqrt(tf.reduce_sum(tf.square(pos_distance)))
            neg_score = tf.sqrt(tf.reduce_sum(tf.square(neg_distance)))
        else:
            raise RuntimeError("只支持L1或L2范数")

        return tf.reduce_sum(tf.maximum(self.margin + pos_score - neg_score, 0))

