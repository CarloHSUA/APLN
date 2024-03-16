# Descargamos la librería FAISS
import faiss
from faiss import write_index
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader


from datasets import load_dataset
# Columns: ['story', 'questions', 'answers', 'summary']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = {}

# Read data base from huggingface
def read_data():
    # TODO: glnmario/news-qa-summarization
    '''
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)
    '''
    dataset = load_dataset("glnmario/news-qa-summarization", split="train")
    print(dataset[0]['questions'])
    print('\n')
    print(dataset[0]['story'])

    # dataset.set_format(type="torch", columns=["story", "questions"])
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=64)
    d = next(iter(dataloader))
    print(d)
    for data in dataloader:
        print(data)
        break


def get_data_chunks():

    for data in dataset:
        pass

def tokenize_dataset(tokenizer, model):
    # Tokenize the story column
    for data in dataset:
        inputs = tokenizer(data['story'], return_tensors='pt', truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)


if __name__ == '__main__':
    read_data()

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # model = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)

    # tokenize_dataset(tokenizer, model)


