# Sentiment Order Study

**Sequence learning models like LSTMs are often assumed to depend heavily on input order. But how much does this matter in practice, and can modern Transformers overcome this limitation?**

This research project explores the effect of training order on LSTM and Transformer models for sentiment analysis.

**Explore the full experiment dashboard with MLflow**: [sentiment-order-lab.onrender.com](https://sentiment-order-lab.onrender.com)  
*🛈 Note: This is a live Render app — if it’s asleep, give it ~1 minute to wake up.*



## 💡 Research Question
> Can LSTM and Transformer-based models produce significantly different sentiment predictions when the training data sequence is altered?

## Overview
This project compares two deep learning architectures — **LSTM** and **TinyBERT** — for sentiment classification on the **IMDb Movie Reviews** dataset. The focus is on testing whether **training data order** impacts the performance of each model.

Both models are trained twice:
1. **Original Order** (default dataset order)
2. **Shuffled Order** (randomized once before training)

The results are tracked using **MLflow**, with accuracy, F1 scores, and confusion matrices compared across both runs.

## Dataset
I use the [IMDb Movie Review dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), containing 50,000 labeled reviews.
- A subset of **20,000 reviews** was selected and split into **training**, **validation**, and **test** sets.
- Binary sentiment labels: `positive` / `negative`.
- Preprocessing:
  - For **LSTM**: Reviews are tokenized and lemmatized.
  - For **TinyBERT**: Raw text is passed directly into the tokenizer (no preprocessing).

## Model Performance
| Model     | Order     | Accuracy | F1 Score |
|-----------|-----------|----------|----------|
| LSTM      | Original  | 80.6%    | 0.82     |
| LSTM      | Shuffled  | 72.7%    | 0.63     |
| TinyBERT  | Original  | 84.6%    | 0.83     |
| TinyBERT  | Shuffled  | 85.0%    | 0.84     |


## Results Summary
**LSTM** shows a significant performance drop when order is removed, while **TinyBERT** remains stable, showcasing robustness due to its attention-based architecture.

> 📝 **Curious why LSTM performs worse on shuffled data while TinyBERT stays consistent?**  
> Explore [`sentiment_research_report.ipynb`](notebooks/sentiment_research_report.ipynb) — a detailed, step-by-step research log that covers training progress, model choices, and in-depth analysis of how data order affects learning.  

![Model Performance Comparison](images/models_comparison.png)

## Learning Behavior During Training
The chart below compares training accuracy across epochs for both models.  
While **TinyBERT** follows a smooth and consistent learning curve regardless of input order, **LSTM** exhibits less stable behavior and diverging patterns — especially under shuffled conditions.

![Training Accuracy Progression](images/train_accuracy_comparison.png)

## Models Architecture
**LSTM**
* Embedding: Pre-trained Word2Vec (frozen)  
* Input: Lemmatized tokens  
* Architecture:
    * Embedding Layer (pre-trained Word2Vec, non-trainable)
    * LSTM Layer (64 units, with dropout and recurrent dropout)
    * Dense Output Layer (Sigmoid activation) 
* Output: Binary classification via Sigmoid  

**TinyBERT**
* Hugging Face pretrained checkpoint
* Input: Untouched raw sentences (tokenized via tokenizer)
* Architecture:
    * Pre-trained Transformer Encoder (TinyBERT General 4L 312D)
    * Dropout Layer
    * Classification Head (Dense Layer) 
* Output: Softmax binary classifier

## Tech Stack
* Python 3.10
* PyTorch
* Hugging Face Transformers
* TensorFlow
* Gensim (Word2Vec)
* Scikit-learn
* MLflow
* Docker + Render (for deployment)

## Repository Structure
```
sentiment-lstm-vs-transformer/  
│  
├── data/  
│   ├── IMDB_Dataset_large.csv  
│   ├── IMDB_Dataset_medium.csv  
│   └── IMDB_Dataset_small.csv  
│  
├── logs/  
│   ├── evaluation_results.log  
│   └── stats.log  
│  
├── mlruns/  
│  
├── models/  
│   ├── LSTM_sentiment_original.h5  
│   ├── LSTM_sentiment_shuffled.h5  
│   ├── TinyBERT_sentiment_original.pt  
│   └── TinyBERT_sentiment_shuffled.pt  
│  
├── notebooks/   
│   └── sentiment_research_report.ipynb  
│  
├── src/  
│   ├── models/  
│   │   ├── lstm.py  
│   │   └── tinybert.py
│   ├── config.py  
│   ├── main.py  
│   ├── preprocess.py  
│   └── stats.py  
│
├── .gitignore  
├── Dockerfile  
├── README.md  
└── requirements.txt  
 ```

## Getting Started
#### 1. Clone the Repo  
```
git clone https://github.com/coralen/sentiment-order-study.git
cd sentiment-order-study
```
#### 2. Install Dependencies  
```
pip install -r requirements.txt
```

#### 3. Run Training
```
python main.py
```
**Total Training Time:** ~6.5 hours (for all 4 training runs on a standard CPU)

if you wish to run a faster and lighter version, change the value of `DATA_PATH` in file `src/config.py` from medium.csv to small.csv. Total training time for the lighter version is about ~5 minutes.

#### 4. Launch MLflow UI Locally (Optional)  
```
mlflow ui --backend-store-uri ./mlruns  
```  
Open http://localhost:5000 to browse the experiment logs.

#### 5. Run the notebook on your own results (Optional)  
*Note: The notebook will need some modifications if you wish to view you own run results.*  
Run mlflow locally. Locate in the notebook the line `mlflow.set_tracking_uri('https://sentiment-order-lab.onrender.com/')`, and change the url to `https://127.0.0.1:5000/`
