# Descargamos la librería FAISS
import faiss
from faiss import write_index
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertForQuestionAnswering
from torch.utils.data import DataLoader

from faiss_wrapper import FaissWrapper
from model.sentence_model import SencenceModel



from datasets import load_dataset
# Columns: ['story', 'questions', 'answers', 'summary']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = {}


# 'google-bert/bert-large-uncased-whole-word-masking-finetuned-squad'
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# model = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)

model = SencenceModel(device = DEVICE)

dataset = load_dataset("glnmario/news-qa-summarization", split="train")
faiss = FaissWrapper('./prueba.index')


# Read data base from huggingface
def read_data():
    global dataset
    # Tokenize text

    # dataset = dataset.map(
    #     lambda element: model.tokenizer(element['story'], padding=True, truncation=True),
    #     batched=True,
    #     batch_size=1
    #     )   

    dataset_tokenized = dataset.remove_columns(['answers', 'summary', 'questions'])# , 'story'])
    dataset_tokenized.set_format("torch")

    dataloader = DataLoader(dataset_tokenized, batch_size=4)

        
    for idx, batch in enumerate(dataloader):
        # batch = {key: value.to(DEVICE) for key, value in batch.items()}  # Mover los tensores al dispositivo CUDA
        # print(batch['story'])
        embeddings = model.calculate_embedding(batch['story'])
        print(embeddings)
        # print(batch, model(**batch))
        # prediction = model(**batch) # bacth x doc x cada token del doc
        # faiss.add_embeddings()
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


