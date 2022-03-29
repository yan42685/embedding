from tf_model.base_model import BaseModel
import tensorflow as tf
from tools import time_it


class TransE(BaseModel):
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

    @time_it
    def _update_embedding(self, pos_quads, neg_quads):
        for pos_quad in pos_quads:
            for neg_quad in neg_quads:
                pos_h, pos_r, pos_t = self._lookup_embedding(pos_quad)
                neg_h, neg_r, neg_t = self._lookup_embedding(neg_quad)
                variables = [pos_h, pos_r, pos_t, neg_h, neg_r, neg_t]

                with tf.GradientTape() as tape:
                    loss = self._loss_function(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
                grads = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
                # print("step loss: %.4f" % loss.numpy())

