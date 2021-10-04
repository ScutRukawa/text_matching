from abc import ABC
import tensorflow as tf
from config.config import ModelConfig

from tensorflow.keras import regularizers

from transformers import TFBertModel, BertTokenizer, BertConfig

cfg = ModelConfig()


class BertMatch(tf.keras.Model, ABC):
    def __init__(self):
        super(BertMatch, self).__init__()
        bert_config = BertConfig.from_json_file('./bert-base-chinese/config.json')
        self.bert_model = TFBertModel.from_pretrained('./bert-base-chinese', config=bert_config)
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese', config=bert_config)
        self.bilstm = tf.keras.layers.LSTM(units=cfg.lstm_hidden_dim, activation='tanh', return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=2, activation=None,
                                           kernel_regularizer=regularizers.l2(0.001))
        self.flatten = tf.keras.layers.Flatten(name='flatten')
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    def call(self, inputs, training=None, mask=None):
        _, cls = self.bert_model(inputs[0], attention_mask=mask, token_type_ids=inputs[1])

        cls = self.layer_norm(cls)
        logits = self.dense(cls)
        pred = tf.nn.softmax(logits)
        return logits, pred
