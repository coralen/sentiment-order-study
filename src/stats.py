from datetime import datetime
from collections import Counter
import json

from config import TIMESTAMP_FORMAT


def calculate_word_counts(data):
    raw_word_count = sum(len(review.split()) for review in data['review'])
    token_count = sum(len(tokens) for tokens in data['tokens'])
    lemma_count = sum(len(lemmas) for lemmas in data['lemmas'])
    
    return {
        'raw': raw_word_count,
        'tokens': token_count,
        'lemmas': lemma_count
    }

def calculate_pos_distribution(data, top_n=10):
    all_pos = [pos for doc in data["pos"] for pos in doc]
    pos_counts = Counter(all_pos)
    return dict(pos_counts.most_common(top_n))

def calculate_length_buckets(data, bucket_size=100, max_bucket=1000):
    token_lengths = [len(tokens) for tokens in data['tokens']]
    buckets = [min((length // bucket_size) * bucket_size, max_bucket) 
               for length in token_lengths]
    bucket_counts = Counter(buckets)
    sorted_buckets = sorted(bucket_counts.items())
    
    bucket_labels = [f"{b}-{b + bucket_size - 1}" if b < max_bucket 
                    else f"{max_bucket}+" for b, _ in sorted_buckets]
    bucket_values = [count for _, count in sorted_buckets]
    
    return {label: value for label, value in zip(bucket_labels, bucket_values)}

def create_stats_dict(data):
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    word_counts = calculate_word_counts(data)
    pos_distribution = calculate_pos_distribution(data)
    length_buckets = calculate_length_buckets(data)
    
    return {
        "timestamp": timestamp,
        **word_counts,
        "pos_count": pos_distribution,
        "len_buckets": length_buckets
    }

def save_stats(stats_dict, log_path):    
    with open(log_path, "a") as f:
        f.write(json.dumps(stats_dict) + "\n")

def collect_stats(data, log_path):
    stats_dict = create_stats_dict(data)
    save_stats(stats_dict, log_path)