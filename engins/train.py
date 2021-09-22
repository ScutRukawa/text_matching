import configparser
import json
from engins.data import DataProcessor
from tqdm import tqdm
from engins.model.esim_model import ESIMModel
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import metrics

from engins.metrics import evaluate
from config.config import ModelConfig

cfg = ModelConfig()


class TrainProcessor:
    def __init__(self):
        super(TrainProcessor, self).__init__()

    def train(self):
        acc_meter = metrics.Accuracy()
        loss_meter = metrics.Mean()

        data_processor = DataProcessor()
        model = ESIMModel()
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=cfg.checkpoints_dir,
                                                        max_to_keep=cfg.max_to_keep,
                                                        checkpoint_name=cfg.checkpoint_name)
        train_data = data_processor.load_data('./data/train.csv')
        dev_data = data_processor.load_data('./data/dev.csv')

        for epoch in range(cfg.epochs):

            for step, batch in tqdm(train_data.batch(cfg.batch_size).enumerate(),
                                    desc='epoch:' + str(epoch)):
                X_train_s1, X_train_s2, y_train, mask1, mask2 = batch
                X = (X_train_s1, X_train_s2)
                mask = (mask1, mask2)
                with tf.GradientTape() as tape:
                    logits, y_pred = model(X, mask=mask, training=True)
                    y = tf.cast(y_train, dtype=tf.int32)
                    y = tf.one_hot(y, depth=2)
                    loss = categorical_crossentropy(y, y_pred)
                    loss = tf.reduce_mean(loss, axis=-1)
                    loss_meter.update_state(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                cfg.optimizers.apply_gradients(zip(grads, model.trainable_variables))

            print("epoch %d :  loss: %f" % (epoch, loss_meter.result().numpy()

                                            ))
            all_pred, all_true = np.array([]), np.array([])
            for step, batch in tqdm(dev_data.batch(cfg.batch_size).enumerate(), desc='epoch:' + str(epoch)):
                X_dev_s1, X_dev_s2, y_dev, dev_mask1, dev_mask2 = batch
                X = (X_dev_s1, X_dev_s2)
                mask = (dev_mask1, dev_mask2)
                logits, y_pred = model(X, mask=mask, training=False)
                # y_dev = tf.one_hot(y_dev, depth=2)

                pred = tf.argmax(y_pred, axis=1)
                acc_meter.update_state(y_dev, pred)

                all_pred = np.append(all_pred, y_pred)
                all_true = np.append(all_true, y_dev)
            acc = float(acc_meter.result().numpy())
            print("epoch %d :  acc: %f" % (epoch, acc))
            checkpoint_manager.save()
