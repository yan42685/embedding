from model.tf_model import TfModel
import tensorflow as tf


class TransE(TfModel):
    def loss_function(self, pos_quad, neg_quad):
        pos_h = tf.nn.embedding_lookup(self.entities_embedding, pos_quad[0])
        pos_r = tf.nn.embedding_lookup(self.relations_embedding, pos_quad[1])
        pos_t = tf.nn.embedding_lookup(self.entities_embedding, pos_quad[2])
        neg_h = tf.nn.embedding_lookup(self.entities_embedding, neg_quad[0])
        neg_r = tf.nn.embedding_lookup(self.relations_embedding, neg_quad[1])
        neg_t = tf.nn.embedding_lookup(self.entities_embedding, neg_quad[2])

        pos_distance = pos_h + pos_r - pos_t
        neg_distance = neg_h + neg_r - neg_t
        if self.norm == "L1":
            pos_score = tf.reduce_sum(tf.abs(pos_distance), axis=1)
            neg_score = tf.reduce_sum(tf.abs(neg_distance), axis=1)
        elif self.norm == "L2":
            pos_score = tf.sqrt(tf.reduce_sum(tf.square(pos_distance), axis=1))
            neg_score = tf.sqrt(tf.reduce_sum(tf.square(neg_distance), axis=1))
        else:
            raise RuntimeError("只支持L1或L2范数")

        return tf.reduce_sum(tf.maximum(self.margin + pos_score - neg_score, 0))
