from dataclasses import dataclass

# Process
TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
DATA_PATH = 'data/IMDB_Dataset_small.csv'

# Log files path
EVAL_LOG_PATH = 'logs/evaluation_results.log'
STATS_LOG_PATH = 'logs/stats.log'

@dataclass
class LSTMConfig:
    max_len: int = 200
    lstm_units: int = 64
    loss: str = 'binary_crossentropy'
    optimizer: str = 'adam'
    dropout: float = 0.2
    recurrent_dropout: float = 0.2
    model_name: str = "LSTM_classifier"
    batch_size: int = 64
    epochs: int = 2
    validation_split: float = 0.2
    patience: int = 5


@dataclass
class TinyBERTConfig:
    max_len: int = 200
    model_name: str = "TinyBERT_classifier"
    learning_rate: float = 2e-5
    epochs: int = 1
    batch_size: int = 64
