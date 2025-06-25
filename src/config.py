from dataclasses import dataclass

# Process
TEST_SIZE = 0.2
RANDOM_STATE = 42
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
DATA_PATH = 'data/IMDB_Dataset_medium.csv'

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
    experiment_name: str = "LSTM_sentiment_experiment"
    batch_size: int = 64
    epochs: int = 10
    validation_split: float = 0.2
    patience: int = 5


@dataclass
class TinyBERTConfig:
    max_len: int = 200
    model_name: str = "TinyBERT_classifier"
    experiment_name: str = "TinyBERT_sentiment_experiment"
    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 64

@dataclass
class Word2VecConfig:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    workers: int = 4