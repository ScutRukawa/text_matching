import numpy as np
import tensorflow as tf
import pandas as pd
from engins.data import DataProcessor
from engins.train import TrainProcessor

if __name__ == '__main__':
    d = DataProcessor()
    trainProcessor = TrainProcessor()
    trainProcessor.train()
    # d.build_voc()
    #
