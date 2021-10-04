from tensorflow.keras import optimizers, metrics


class ModelConfig:
    def __init__(self):
        self.max_seq_length = 100
        self.class_num = 2
        self.voc_size = 8180
        self.batch_size = 8
        self.is_early_stop = True
        self.patient = 5
        self.embedding_method = 'embedding'
        self.embedding_dim = 300
        self.lstm_hidden_dim = 32
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        self.optimizers = optimizers.Adagrad()
        self.max_to_keep = 3
        self.checkpoints_dir = './model/esim'
        self.checkpoint_name = 'esim'
        self.epochs = 10
        self.use_bert = False
        self.seg = ['[SEG]']
