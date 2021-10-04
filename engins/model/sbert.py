from abc import ABC
import tensorflow as tf
from config.config import ModelConfig

from tensorflow.keras import regularizers

from transformers import TFBertModel, BertTokenizer, BertConfig

cfg = ModelConfig()


class SBertMatch(tf.keras.Model, ABC):
    def __init__(self):
        super(SBertMatch, self).__init__()
        bert_config = BertConfig.from_json_file('./bert-base-chinese/config.json')
        self.bert_model = TFBertModel.from_pretrained('./bert-base-chinese', config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese', config=bert_config)
        self.bilstm = tf.keras.layers.LSTM(units=cfg.lstm_hidden_dim, activation='tanh', return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=2, kernel_initializer='he_normal',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.drop_out = tf.keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=None, mask=None):
        hidden_states_1, cls1 = self.bert_model(inputs=inputs[0], attention_mask=mask[0])
        hidden_states_2, cls2 = self.bert_model(inputs=inputs[1], attention_mask=mask[1])
        # cls1 = self.layer_norm(cls1)
        # cls2 = self.layer_norm(cls2)
        cls1 = tf.math.reduce_mean(hidden_states_1, 1)
        cls2 = tf.math.reduce_mean(hidden_states_2, 1)
        concat_embedding = tf.concat([cls1, cls2, tf.math.abs(cls1 - cls2)], axis=1)
        drop_out = self.drop_out(concat_embedding)

        logits = self.dense(drop_out)
        pred = tf.nn.softmax(logits)
        return logits, pred
