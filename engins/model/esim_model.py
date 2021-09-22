from abc import ABC
from config.config import ModelConfig

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Embedding, Attention, Dropout, Dense

from config.config import ModelConfig

cfg = ModelConfig()


class ESIMModel(tf.keras.Model, ABC):
    def __init__(self):
        super(ESIMModel, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(units=cfg.lstm_hidden_dim, return_sequences=True))
        self.bi_lstm2 = Bidirectional(LSTM(units=cfg.lstm_hidden_dim, return_sequences=True))

        self.embedding = Embedding(input_dim=cfg.voc_size, output_dim=cfg.embedding_dim,
                                   input_length=cfg.max_seq_length)
        self.dropout = Dropout(rate=0.5)
        self.dense = Dense(units=2, activation=None)

    def soft_align_attention(self, s1, s2, mask1, mask2):
        # masked_s1 = tf.ragged.boolean_mask(s1, mask1)  # [1, 3]
        # masked_s2 = tf.ragged.boolean_mask(s2, mask2)

        attention = s1 @ tf.transpose(s2, [0, 2, 1])
        weight_1 = tf.nn.softmax(attention, axis=0)
        weight_2 = tf.nn.softmax(tf.transpose(attention, perm=[0, 2, 1]))

        s1_align = weight_1 @ s2
        s2_align = weight_2 @ s1

        return s1_align, s2_align

    # def submul(self, s1, s2):
    #     sub = s1 - s2
    #     mul = s1 * s2
    #     return sub, mul

    def pooling(self, s):
        avg_pool = tf.reduce_mean(s, axis=1)
        max_pool = tf.reduce_max(s, axis=1)
        return tf.concat([avg_pool, max_pool], axis=1)

    def call(self, inputs, training=None, mask=None):
        embedding_a = self.embedding(inputs[0])
        embedding_b = self.embedding(inputs[1])

        encode_a = self.bi_lstm(embedding_a)
        encode_b = self.bi_lstm(embedding_b)

        a_align, b_align = self.soft_align_attention(encode_a, encode_b, mask[0], mask[1])

        combine_a = tf.concat([encode_a, a_align, encode_a - a_align, encode_a * a_align], axis=2)
        combine_b = tf.concat([encode_b, b_align, encode_b - b_align, encode_b * b_align], axis=2)

        compose_a = self.bi_lstm2(combine_a)
        compose_b = self.bi_lstm2(combine_b)

        pool_a = self.pooling(compose_a)
        pool_b = self.pooling(compose_b)

        x = tf.concat([pool_a, pool_b], axis=1)
        x = self.dropout(x)

        logits = self.dense(x)

        prob = tf.nn.softmax(logits, axis=1)
        return logits, prob
