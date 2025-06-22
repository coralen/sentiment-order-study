from sklearn.model_selection import train_test_split
import pandas as pd
import os
import spacy

class Preprocess:
    def __init__(self, file_path=None, test_size: float=0.2, batch_size=256):
        self.train_data = None
        self.test_data = None
        self.file_path = file_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
    def load_and_split(self):
        if self.file_path is None:
            self.file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'IMDB_Dataset_sample.csv')
            self.file_path = os.path.abspath(self.file_path)
        
        try:
            imdb_df = pd.read_csv(self.file_path)
            imdb_df['label'] = imdb_df['sentiment'].map({'positive': 1, 'negative': 0})
            self.train_data, self.test_data = train_test_split(imdb_df, test_size=0.2, random_state=42)
        except Exception as e:
            raise

    def extract_features(self, docs):
        tokens = []
        lemmas = []
        pos_tags = []
        for doc in docs:
            tokens.append([t.text for t in doc if t.is_alpha])
            lemmas.append([t.lemma_ for t in doc if t.is_alpha])
            pos_tags.append([t.pos_ for t in doc if t.is_alpha])
        return tokens, lemmas, pos_tags

    def nlp_pipeline(self, data, type):
        data_docs = self.nlp.pipe(data["review"], batch_size=self.batch_size, n_process=1)
        data["tokens"], data["lemmas"], data["pos"] = self.extract_features(data_docs)
        if type=="train": 
            self.train_data = data
        else: 
            self.test_data = data
    
    def process(self):
        self.load_and_split()
        self.nlp_pipeline(self.train_data, type="train")
        self.nlp_pipeline(self.test_data, type="test")
        return self.train_data, self.test_data
