import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import numpy as np
import mlflow
import json

from config import TIMESTAMP_FORMAT


class LSTMClassifier:
    def __init__(self, config):
        self.model = None
        self.lstm_units = config.lstm_units
        self.loss = config.loss
        self.optimizer = config.optimizer
        self.dropout = config.dropout
        self.recurrent_dropout = config.recurrent_dropout
        self.model_name = config.model_name
        self.experiment_name = config.experiment_name
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.validation_split = config.validation_split
        self.patience = config.patience

    def build_model(self, embedding_matrix):      
        self.model = Sequential()
        self.model.add(Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=False
        ))
        self.model.add(LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

    def train(self, train_padded, train_labels, embedding_matrix):
        self.build_model(embedding_matrix)
        early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.model_name) as run:
            # Log parameters
            mlflow.log_param("lstm_units", self.lstm_units)
            mlflow.log_param("embedding_dim", embedding_matrix.shape[1])
            mlflow.log_param("vocab_size", embedding_matrix.shape[0])
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("validation_split", self.validation_split)

            # Train model
            for epoch in range(self.epochs):
                history = self.model.fit(
                    train_padded,
                    train_labels,
                    epochs=1,
                    batch_size=self.batch_size,
                    validation_split=self.validation_split,
                    callbacks=([early_stop]),
                    verbose=1
                )

                # Log metrics for this epoch
                mlflow.log_metric("train_loss", history.history["loss"][0], step=epoch)
                mlflow.log_metric("val_loss", history.history["val_loss"][0], step=epoch)
                mlflow.log_metric("train_accuracy", history.history["accuracy"][0], step=epoch)
                mlflow.log_metric("val_accuracy", history.history["val_accuracy"][0], step=epoch)

        return run.info.run_id
    
    def evaluate(self, test_data, test_labels, threshold=0.5, log=True, log_path=None, run_id=None):
        # Get predictions
        probabilities = self.model.predict(test_data)
        predictions = (probabilities > threshold).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        conf_matrix = confusion_matrix(test_labels, predictions)
        
        # Keras evaluation for loss
        test_loss, _ = self.model.evaluate(test_data, test_labels, verbose=0)
        
        if run_id is not None:
            mlflow.start_run(run_id=run_id)
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.end_run()
        
        if log:
            results = {
                "timestamp": datetime.now().strftime(TIMESTAMP_FORMAT),
                "model_name": self.model_name,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "f1_score": f1,
                "confusion_matrix": np.array2string(conf_matrix, separator=', ')
            }
            
            with open(log_path, "a") as f:
                f.write(json.dumps(results) + "\n")
    
    def save(self):
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{self.model_name}.h5")
        self.model.save(model_path)
        print(f"LSTM model saved to {model_path}")
        