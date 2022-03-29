from model.tf_model import TfModel
import tensorflow as tf


class TransE(TfModel):
    def _loss_function(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
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

    def _update_embedding(self, pos_quads, neg_quads):
        for pos_quad in pos_quads:
            for neg_quad in neg_quads:
                pos_h = tf.nn.embedding_lookup(self.entities_embedding, pos_quad[0])
                pos_r = tf.nn.embedding_lookup(self.relations_embedding, pos_quad[1])
                pos_t = tf.nn.embedding_lookup(self.entities_embedding, pos_quad[2])
                neg_h = tf.nn.embedding_lookup(self.entities_embedding, neg_quad[0])
                neg_r = tf.nn.embedding_lookup(self.relations_embedding, neg_quad[1])
                neg_t = tf.nn.embedding_lookup(self.entities_embedding, neg_quad[2])
                variables = [pos_h, pos_r, pos_t, neg_h, neg_r, neg_t]

                with tf.GradientTape() as tape:
                    loss = self._loss_function(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
                grads = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
                print("step loss: %.4f" % loss.numpy())
