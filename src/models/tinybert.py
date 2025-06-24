import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datetime import datetime
import numpy as np
import torch
import mlflow
import json


class TinyBERTClassifier:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained('huawei-noah/TinyBERT_General_4L_312D', 
                                                                   num_labels=2).to(self.device)
        self.model_name = config.model_name
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        
    def train(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        mlflow.set_experiment("sentiment_experiment_TinyBERT")
        with mlflow.start_run(run_name=self.model_name) as run:
            mlflow.log_params({
                "learning_rate": self.learning_rate,
                "epochs": self.epochs
            })

            for epoch in range(self.epochs):               
                total_loss = 0
                correct = 0
                total = 0
                
                self.model.train()

                for batch in train_loader:
                    input_ids, attention_mask, labels = [x.to(self.device) for x in batch]

                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                avg_loss = total_loss / len(train_loader)
                accuracy = correct / total

                val_acc, val_loss, _, _ = self.evaluate(val_loader, log=False)

                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                }, step=epoch)

                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        return run.info.run_id

    def evaluate(self, dataloader, log=True, log_path=None, run_id=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                total_loss += loss.item()

                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        avg_loss = total_loss / len(dataloader)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        if run_id is not None:
            mlflow.start_run(run_id=run_id)
            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("test_loss", avg_loss)
            mlflow.end_run()

        if log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            results = {
                "timestamp": timestamp,
                "model_name": self.model_name,
                "test_loss": avg_loss,
                "test_accuracy": acc,
                "f1_score": f1,
                "confusion_matrix": np.array2string(conf_matrix, separator=', ')
            }
            
            with open(log_path, "a") as f:
                f.write(json.dumps(results) + "\n")

        return acc, avg_loss, f1, conf_matrix

    def save(self):
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{self.model_name}.pt")
        torch.save(self.model.state_dict(), model_path)
        print(f"TinyBERT model saved to {model_path}")