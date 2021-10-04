from engins.data import DataProcessor
from engins.train import TrainProcessor
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    d = DataProcessor()
    trainProcessor = TrainProcessor()
    trainProcessor.train()
    d.build_voc()
    # hidden_states_1 = np.random.random([1, 2, 3])
    # cls1 = tf.math.reduce_mean(hidden_states_1, 1)
    # print(cls1)
