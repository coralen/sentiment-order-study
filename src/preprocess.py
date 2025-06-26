import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import torch
import spacy

from config import TEST_SIZE, RANDOM_STATE

    
def load_and_split(file_path):
    try:
        imdb_df = pd.read_csv(file_path)
        imdb_df['label'] = imdb_df['sentiment'].map({'positive': 1, 'negative': 0})
        return train_test_split(imdb_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    except Exception as e:
        raise

def extract_features(docs):
    tokens = []
    lemmas = []
    pos_tags = []
    for doc in docs:
        tokens.append([t.text for t in doc if t.is_alpha])
        lemmas.append([t.lemma_ for t in doc if t.is_alpha])
        pos_tags.append([t.pos_ for t in doc if t.is_alpha])
    return tokens, lemmas, pos_tags

def nlp_pipeline(df):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    data_docs = nlp.pipe(df["review"])
    df["tokens"], df["lemmas"], df["pos"] = extract_features(data_docs)
    return df

def process(file_path):
    '''Process text for tokens, lemmas and pos tags, and create a shuffled train set'''
    train_df, test_df = load_and_split(file_path)
    train_df = nlp_pipeline(train_df)
    test_df = nlp_pipeline(test_df)
    train_df_shuffled = shuffle(train_df, random_state=RANDOM_STATE)
    return train_df, train_df_shuffled, test_df

def tinybert_tokenizer(tokenizer, df, max_len):
    return tokenizer.batch_encode_plus(
            df.tolist(),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='pt'
        )

def prepare_data_tinybert(train_df, train_labels, test_df, config):
    '''Accept raw data since TinyBERT works better with it.'''
    tokenizer = BertTokenizerFast.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    X_train, X_val, y_train, y_val = train_test_split(train_df, train_labels, test_size=TEST_SIZE, 
                                                      random_state=RANDOM_STATE)

    train_tokens = tinybert_tokenizer(tokenizer, X_train['review'], config.max_len)
    train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], 
                                    torch.tensor(y_train.tolist()))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    val_tokens = tinybert_tokenizer(tokenizer, X_val['review'], config.max_len)
    val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'], 
                                torch.tensor(y_val.tolist()))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    test_tokens = tinybert_tokenizer(tokenizer, test_df['review'], config.max_len)
    test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], 
                                    torch.tensor(test_df['label'].tolist()))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader

def prepare_data_lstm(train_texts, test_texts, max_len, w2v_config):
    '''Accept lemmas since LSTM works better with it.''' 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Pad sequences
    train_padded = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

    w2v_model = Word2Vec(sentences=train_texts, vector_size=w2v_config.vector_size, 
                         window=w2v_config.window, min_count=w2v_config.min_count, workers=w2v_config.workers)

    # Create embedding matrix
    word_index = tokenizer.word_index
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]

    return train_padded, test_padded, embedding_matrix
