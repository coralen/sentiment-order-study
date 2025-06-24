from gensim.models import Word2Vec
from sklearn.utils import shuffle

from models.lstm import LSTMClassifier
from models.tinybert import TinyBERTClassifier
from config import EVAL_LOG_PATH, STATS_LOG_PATH, DATA_PATH, LSTMConfig, TinyBERTConfig
import preprocess
import stats


def main():
    # Process text for tokens, lemmas and pos tags
    train_texts, test_texts = preprocess.process(DATA_PATH)
    
    train_labels = train_texts['label'].values
    train_lemmas = train_texts['lemmas']

    test_labels = test_texts['label'].values
    test_lemmas = test_texts['lemmas']

    train_texts_shuffled = shuffle(train_texts, random_state=42)
    train_labels_shuffled = train_texts_shuffled['label'].values
    train_lemmas_shuffled = train_texts_shuffled['lemmas']
    
    # Collect statistics
    stats.log_stats(train_texts, STATS_LOG_PATH)

    # Word2Vec
    w2v_original = Word2Vec(sentences=train_lemmas, vector_size=100, window=5, min_count=2, workers=4)
    w2v_shuffled = Word2Vec(sentences=train_lemmas_shuffled, vector_size=100, window=5, min_count=2, workers=4)

    # Process, train and evaluate original LSTM model
    LSTMConfig.model_name = "LSTM_sentiment_original"
    lstm_original = LSTMClassifier(LSTMConfig)
    train_padded, test_processed, embedding_matrix = preprocess.prepare_data_lstm(train_lemmas, test_lemmas, 
                                                                                  w2v_original, LSTMConfig.max_len)
    mlflow_original = lstm_original.train(train_padded, train_labels, embedding_matrix)
    lstm_original.evaluate(test_processed, test_labels, log_path=EVAL_LOG_PATH, run_id=mlflow_original)
    lstm_original.save()

    # Process, train and evaluate shuffled LSTM model
    LSTMConfig.model_name = "LSTM_sentiment_shuffled"
    lstm_shuffled = LSTMClassifier(LSTMConfig)
    train_padded_shuffled, _, embedding_matrix_shuffled = preprocess.prepare_data_lstm(train_lemmas_shuffled, 
                                                                    test_lemmas, w2v_shuffled, LSTMConfig.max_len)
    mlflow_shuffled = lstm_shuffled.train(train_padded_shuffled, train_labels_shuffled, embedding_matrix_shuffled)
    lstm_shuffled.evaluate(test_processed, test_labels, log_path=EVAL_LOG_PATH, run_id=mlflow_shuffled)
    lstm_shuffled.save()

    print(EVAL_LOG_PATH)
    # Process, train and evaluate original TinyBERT model
    TinyBERTConfig.model_name = "TinyBERT_sentiment_original"
    tinybert_original = TinyBERTClassifier(TinyBERTConfig)
    train_loader, val_loader, test_loader = preprocess.prepare_data_tinybert(train_texts, train_labels, 
                                                                             test_texts, TinyBERTConfig)
    bert_mlflow_original = tinybert_original.train(train_loader, val_loader)
    tinybert_original.evaluate(test_loader, log_path=EVAL_LOG_PATH, run_id=bert_mlflow_original)
    tinybert_original.save()

    # Process, train and evaluate shuffled TinyBERT model
    TinyBERTConfig.model_name = "TinyBERT_sentiment_shuffled"
    tinybert_shuffled = TinyBERTClassifier(TinyBERTConfig)
    train_loader_shuffled, val_loader_shuffled, _ = preprocess.prepare_data_tinybert(train_texts_shuffled, 
                                                        train_labels_shuffled, test_texts, TinyBERTConfig)
    bert_mlflow_shuffled = tinybert_shuffled.train(train_loader_shuffled, val_loader_shuffled)
    tinybert_shuffled.evaluate(test_loader, log_path=EVAL_LOG_PATH, run_id=bert_mlflow_shuffled)
    tinybert_shuffled.save()

if __name__ == '__main__':
    main()