# LSTM
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# evaluation
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import mlflow
import numpy as np

class LSTMClassifier:
    def __init__(self, max_len=200, lstm_units=64, 
                 loss='binary_crossentropy', optimizer='adam', model_name="LSTM_classifier"):
        self.max_len = max_len
        self.lstm_units = lstm_units
        self.loss = loss
        self.optimizer = optimizer
        self.model_name = model_name
        self.model = None
        
        # These will be set during training
        self.tokenizer = None
        self.model = None
        self.embedding_matrix = None

    def prepare_data(self, train_texts, test_texts, w2v_model):
        """
        Tokenize and pad text data, create embedding matrix.
        """
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(train_texts)
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)

        # Pad sequences
        train_padded = pad_sequences(train_sequences, maxlen=self.max_len, padding='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_len, padding='post')

        # Create embedding matrix
        word_index = self.tokenizer.word_index
        embedding_dim = w2v_model.vector_size
        self.embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            if word in w2v_model.wv:
                self.embedding_matrix[i] = w2v_model.wv[word]

        return train_padded, test_padded

    def build_model(self):
        """
        Builds the LSTM model using the given embedding matrix.
        """
        if self.embedding_matrix is None:
            raise ValueError("Must call prepare_data() before building model")
            
        self.model = Sequential()
        self.model.add(Embedding(
            input_dim=self.embedding_matrix.shape[0],
            output_dim=self.embedding_matrix.shape[1],
            weights=[self.embedding_matrix],
            trainable=False
        ))
        self.model.add(LSTM(self.lstm_units, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

    def train(self, train_texts, train_labels, test_texts, w2v_model, 
              epochs=10, batch_size=128, validation_split=0.2, 
              early_stopping=True, patience=3):
        """
        Train the LSTM model.
        """
        # Prepare data
        train_padded, test_padded = self.prepare_data(train_texts, test_texts, w2v_model)
        
        # Build model
        self.build_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        mlflow.set_experiment("sentiment_experiment")
        # run with mlflow
        with mlflow.start_run(run_name=self.model_name):
            # Log parameters
            mlflow.log_param("max_len", self.max_len)
            mlflow.log_param("lstm_units", self.lstm_units)
            mlflow.log_param("embedding_dim", self.embedding_matrix.shape[1])
            mlflow.log_param("vocab_size", self.embedding_matrix.shape[0])
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("validation_split", validation_split)

            # Train model
            for epoch in range(epochs):
                history = self.model.fit(
                    train_padded,
                    train_labels,
                    epochs=1,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=([early_stop]),
                    verbose=1
                )

                # Log metrics for this epoch
                mlflow.log_metric("train_loss", history.history["loss"][0], step=epoch)
                mlflow.log_metric("val_loss", history.history["val_loss"][0], step=epoch)
                mlflow.log_metric("train_accuracy", history.history["accuracy"][0], step=epoch)
                mlflow.log_metric("val_accuracy", history.history["val_accuracy"][0], step=epoch)

            # Log metrics
            final_val_accuracy = max(history.history['val_accuracy'])
            final_val_loss = min(history.history['val_loss'])
            
            mlflow.log_metric("final_val_accuracy", final_val_accuracy)
            mlflow.log_metric("final_val_loss", final_val_loss)
            
            # Log model
            mlflow.keras.log_model(self.model, self.model_name)

        return history, test_padded
    
    def predict(self, texts):
        """Make predictions on new text data."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model must be trained before making predictions")
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return self.model.predict(padded)
    
    def evaluate(self, test_data, test_labels, threshold=0.5, print_results=True):
        """
        Detailed evaluation with multiple metrics.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        probabilities = self.model.predict(test_data)
        predictions = (probabilities > threshold).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        conf_matrix = confusion_matrix(test_labels, predictions)
        
        # Keras evaluation for loss
        test_loss, test_acc_keras = self.model.evaluate(test_data, test_labels, verbose=0)
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'test_loss': test_loss,
            'keras_accuracy': test_acc_keras,
            'predictions': predictions,
            'probabilities': probabilities.flatten(),
            'threshold': threshold
        }
        
        if print_results:
            print(f"\n=== LSTM Model Evaluation Results ===")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Classification Threshold: {threshold}")
            print(f"\nConfusion Matrix:")
            print(conf_matrix)
        
        return results