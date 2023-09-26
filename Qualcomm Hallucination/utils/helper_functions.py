from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import json
import random
import math
import jsonlines
import re
import nltk
import pandas as pd
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter1d
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split

class BERTModel:

    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_embedding(self, sentence: str):

        inputs = self.tokenizer(sentence, return_tensors = "pt", truncation = True, max_length = 128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs),
            last_hidden_state = outputs.last_hidden_state
            cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding
    
    def check_similarity(self, sentence1, sentence2):
        inputs_1 = self.tokenizer(sentence1, return_tensors="pt", truncation=True).to(self.device)
        inputs_2 = self.tokenizer(sentence2, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs_1 = self.model(**inputs_1)
            outputs_2 = self.model(**inputs_2)

            embedding_1 = outputs_1.last_hidden_state[:, 0, :]
            embedding_2 = outputs_2.last_hidden_state[:, 0, :]

        cosine_similarity = torch.nn.functional.cosine_similarity(embedding_1, embedding_2).item()

        return cosine_similarity
    
class DatasetFetcher:

    def __init__(self, database_path, train_size, test_size, random_seed = 43, max_prompt_length = 500):

        self.max_prompt_length = max_prompt_length
        self._dataset = pd.read_json(database_path, lines=True)
        self._initialize_pool(train_size, test_size)

    def _initialize_pool(self, train_size, test_size):

        self._pool = self._dataset[self._dataset['instruction'].str.len() <= self.max_prompt_length]
        self._pool = self._pool[self._pool['context'] == '']

        self._pool.reset_index(drop = True, inplace = True)
        matching_indices = self._pool.merge(self._training_pool, how='inner').index
        remaining_pool = self._pool.drop(index = matching_indices).reset_index(drop=True)

        self._testing_pool = remaining_pool.groupby('category').apply(lambda x : x.head(test_size)).reset_index(drop=True)

    def get_examples_from_training_set(self, num_examples):

        if self._train_data.shape[0] < num_examples:
            print("Warning, number of available training examples is less than the requested number, returning all available examples")
            num_examples = self._train_data.shape[0]

        examples = self._train_data.sample(n = num_examples)

        initial_prompts = examples['instruction'].tolist()
        golden_outputs = examples['response'].tolist()
        categories = examples['category'].tolist()

        return initial_prompts, golden_outputs, categories
    

def training_plot(metric, label, filename, num_episodes, agent_lr, agent_action_space, top_k, learning_interval, sentence_dropoff, discount_factor):

    plt.figure(figsize = (9,6))
    plt.plot(metric, color='b', alpha = 5, label = f"Actual {label}")

    smoothing = 1
    metric = np.array(metric)
    if len(metric) < 5:
        metric_smooth = metric
    else:
        metric_smooth = gaussian_filter1d(metric, smoothing)
    
    confidence_interval = stats.sem(metric)

    if label.lower() == "rewards":
        avg_metric = np.mean(metric)
        plt.plot(range(len(metric_smooth)), metric_smooth, label= f"smoothed {label}, AvgReward = {avg_metric.round(3)}", color="red")
    else:
        plt.plot(range(len(metric_smooth)), metric_smooth, label=f"Experiment", color='b')

    plt.fill_between(range(len(metric_smooth)), (metric_smooth - confidence_interval), (metric_smooth + confidence_interval), color="red", alpha=.1)

    params = f'num_episodes = {num_episodes}\n' \
            f'agent_lr = {agent_lr}\n' \
            f"+ Other metrics"
    
    plt.text(1.02, 0.5, params, transform = plt.gca().transAxes, fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='weat', alpha=0.5))
    plt.xlabel('Episodes')
    plt.ylabel(label)
    plt.title(f"{label} vs Episodes")
    plt.subplots_adjust(right=0.7)
    plt.legend(log='upper_right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()