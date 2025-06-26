from config import EVAL_LOG_PATH, STATS_LOG_PATH, DATA_PATH, LSTMConfig, TinyBERTConfig, Word2VecConfig
from models.lstm import LSTMClassifier
from models.tinybert import TinyBERTClassifier
import preprocess
import stats


def handle_lstm_model(LSTMConfig, train_df, test_df):
    train_labels = train_df['label'].values
    train_lemmas = train_df['lemmas']
    test_labels = test_df['label'].values
    test_lemmas = test_df['lemmas']

    lstm_mocel = LSTMClassifier(LSTMConfig)
    train_padded, test_processed, embedding_matrix = preprocess.prepare_data_lstm(train_lemmas, test_lemmas, 
                                                                                  LSTMConfig.max_len, Word2VecConfig)
    mlflow_original = lstm_mocel.train(train_padded, train_labels, embedding_matrix)
    lstm_mocel.evaluate(test_processed, test_labels, log_path=EVAL_LOG_PATH, run_id=mlflow_original)
    lstm_mocel.save()

def handle_tinybert_model(TinyBERTConfig, train_df, test_df):
    train_labels = train_df['label'].values

    tinybert_original = TinyBERTClassifier(TinyBERTConfig)
    train_loader, val_loader, test_loader = preprocess.prepare_data_tinybert(train_df, train_labels, 
                                                                             test_df, TinyBERTConfig)
    bert_mlflow_original = tinybert_original.train(train_loader, val_loader)
    tinybert_original.evaluate(test_loader, log_path=EVAL_LOG_PATH, run_id=bert_mlflow_original)
    tinybert_original.save()

def main():
    # Process text for tokens, lemmas and pos tags
    train_df, train_df_shuffled, test_df = preprocess.process(DATA_PATH)
    
    # Collect statistics
    stats.collect_stats(train_df, STATS_LOG_PATH)

    # Process, train and evaluate LSTM models
    LSTMConfig.model_name = "LSTM_sentiment_original"
    handle_lstm_model(LSTMConfig, train_df, test_df)

    LSTMConfig.model_name = "LSTM_sentiment_shuffled"
    handle_lstm_model(LSTMConfig, train_df_shuffled, test_df)

    # Process, train and evaluate TinyBERT models
    TinyBERTConfig.model_name = "TinyBERT_sentiment_original"
    handle_tinybert_model(TinyBERTConfig, train_df, test_df)

    TinyBERTConfig.model_name = "TinyBERT_sentiment_shuffled"
    handle_tinybert_model(TinyBERTConfig, train_df_shuffled, test_df)

if __name__ == '__main__':
    main()
    