import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd
import csv
import configparser
from transformers import TFBertModel, BertTokenizer, BertConfig
import jieba
from config.config import ModelConfig

cfg = ModelConfig()


class DataProcessor():
    def __init__(self):
        super(DataProcessor, self).__init__()
        token2id_data = pd.read_csv('./data/token2id.csv', sep=' ', quoting=csv.QUOTE_NONE)
        self.token2id = dict(zip(list(token2id_data.token), list(token2id_data.id)))
        self.id2token = dict(zip(list(token2id_data.id), list(token2id_data.token)))

    def build_voc(self):
        train_data = pd.read_csv('./data/train.csv', sep='-', names=['s1', 's2', 'label'])
        token2id = {}
        count = 0
        with open('./data/token2id.csv', 'w', encoding='utf-8') as token2id_file:
            token2id_file.write('token id\n')
            for s1, s2 in zip(list(train_data.s1), list(train_data.s2)):
                for token1, token2 in zip(str(s1).strip(), str(s2).strip()):
                    token2id[token1] = 0
                    token2id[token2] = 0
            for token in token2id.keys():
                token2id_file.write(token + ' ' + str(count) + '\n')
                count += 1
            token2id_file.write('[PAD] ' + str(count) + '\n')
            token2id_file.write('[UNK] ' + str(count + 1) + '\n')

    def sentence_to_vector(self, sentence):
        sentence_ids = []
        for token in sentence:
            if token in self.token2id:
                sentence_ids.append(self.token2id[token])
            else:
                sentence_ids.append(self.token2id[cfg.UNKNOWN])
        sentence_ids = np.array(sentence_ids)
        return sentence_ids

    def load_data(self, file_name):
        count = 0
        sentences_1 = []
        sentences_2 = []
        labels = []
        masks_1 = []
        masks_2 = []
        train_data = pd.read_csv(file_name, sep='-', names=['s1', 's2', 'label'])
        for s1, s2, label in zip(list(train_data.s1), list(train_data.s2), list(train_data.label)):
            if count > 30000:
                break
            count += 1
            if label != '1' and label != '0':
                continue
            s1 = self.processing_sentence(s1)
            s1_padding, mask1 = self.padding_mask(s1)
            s1_ids = self.sentence_to_vector(s1_padding)
            s2 = self.processing_sentence(s2)

            s2_padding, mask2 = self.padding_mask(s2)
            s2_ids = self.sentence_to_vector(s2_padding)
            sentences_1.append(s1_ids)
            sentences_2.append(s2_ids)
            masks_1.append(mask1)
            masks_2.append(mask2)
            labels.append(int(label))
        data_set = tf.data.Dataset.from_tensor_slices(
            (np.array(sentences_1), np.array(sentences_2), np.array(labels), np.array(masks_1), np.array(masks_2)))

        return data_set

    def padding_mask(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < cfg.max_seq_length:
            mask = [True] * len(sentence) + [False] * (cfg.max_seq_length - len(sentence))
            sentence += [cfg.PADDING] * (cfg.max_seq_length - len(sentence))
        else:
            sentence = sentence[:cfg.max_seq_length]
            mask = [True] * cfg.max_seq_length
        return sentence, mask

    def processing_sentence(self, x, stop_words=None):
        cut_word = str(x).strip()
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words
