from gensim.models import Word2Vec
import numpy as np

# utils
from sklearn.utils import shuffle

# local
from preprocess import Preprocess
from models.lstm import LSTMClassifier
from config import LOSS, OPTIMIZER, MAX_LEN

def main():
    # load and process
    preprocessor = Preprocess()
    train_texts, test_texts = preprocessor.process()
    print("Load and process texts")

    # max len for padding
    token_lengths = [len(tokens) for tokens in train_texts['tokens']]
    max_len = int(np.percentile(token_lengths, 85))
    max_len = int(np.ceil(max_len / 100.0)) * 100

    train_texts_shuffled = shuffle(train_texts, random_state=42)

    # Word2Vec
    w2v_model = Word2Vec(sentences=train_texts['tokens'], vector_size=100, window=5, min_count=2, workers=4)
    w2v_model_shuffled = Word2Vec(sentences=train_texts_shuffled['tokens'], vector_size=100, window=5, min_count=2, workers=4)
    print("Trained Word2Vec")

    # Create LSTM classifier - ordered
    lstm_classifier_ordered = LSTMClassifier(
        max_len=MAX_LEN,
        lstm_units=64,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model_name="LSTM_sentiment_classifier_ordered"
    )

    # Create LSTM classifier - shuffled
    lstm_classifier_shuffled = LSTMClassifier(
        max_len=MAX_LEN,
        lstm_units=64,
        loss=LOSS,
        optimizer=OPTIMIZER,
        model_name="LSTM_sentiment_classifier_shuffled"
    )

    print("Created models")

    train_labels = train_texts['label'].values
    train_labels_shuffled = train_texts_shuffled['label'].values
    test_labels = test_texts['label'].values

    # Train ordered model
    history, test_processed = lstm_classifier_ordered.train(
        train_texts=train_texts['tokens'],
        train_labels=train_labels,
        test_texts=test_texts['tokens'],
        w2v_model=w2v_model,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        early_stopping=True,
        patience=5
    )

    # Train shuffled model
    history, _ = lstm_classifier_shuffled.train(
        train_texts=train_texts_shuffled['tokens'],
        train_labels=train_labels_shuffled,
        test_texts=test_texts['tokens'],
        w2v_model=w2v_model_shuffled,
        epochs=10,
        batch_size=64,
        validation_split=0.2,
        early_stopping=True,
        patience=5
    )

    # Evaluate the models
    results_ordered = lstm_classifier_ordered.evaluate(test_processed, test_labels)
    results_shuffled = lstm_classifier_shuffled.evaluate(test_processed, test_labels)

if __name__ == '__main__':
    main()