# Descargamos la librería FAISS
import faiss
from faiss import write_index
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BertForQuestionAnswering
from torch.utils.data import DataLoader
from rich import print
from tqdm import tqdm

from faiss_wrapper import FaissWrapper
from model.sentence_model import SencenceModel
# from preprocess_dataset import load_corupus_dataset



from datasets import load_dataset
# Columns: ['story', 'questions', 'answers', 'summary']

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset = {}


model = SencenceModel(device = DEVICE)

dataset = load_dataset("glnmario/news-qa-summarization", split="train")
faiss_db = FaissWrapper('prueba.index')


# Read data base from huggingface
def read_data():
    global dataset

    dataset_tokenized = dataset.remove_columns(['answers', 'summary', 'questions'])
    dataset_tokenized.set_format("torch")


    dataloader = DataLoader(dataset_tokenized, batch_size=1)
    sentence_list = []


    for idx, batch in tqdm(enumerate(dataloader), ncols=60, desc="Procesando elementos"):
        document = model.nlp(batch['story'][0])
        sentences = list(map(lambda e: str(e), document.sents))
        embeddings = model.calculate_embedding(sentences)
        

        for sentence_idx, sentence in enumerate(list(document.sents)):
            new_row = {}
            new_row['doc_index'] = idx
            new_row['index'] = len(sentence_list) + 1
            new_row['sentence'] = sentence
            new_row['embeddings'] = str(embeddings[sentence_idx].cpu().numpy()).replace('\n', '')
            sentence_list.append(new_row)
        


    sentence_dataset = pd.DataFrame(sentence_list, columns=['doc_index', 'index', 'sentence', 'embeddings'])
    sentence_dataset.to_csv('sentence_dataset.csv')
        


def write_on_faiss():
    data = pd.read_csv('sentence_dataset.csv')
    faiss_db.add_embeddings(data)

    if faiss_db.save_to_disk():
        print("The data is saved correctly")
    


if __name__ == '__main__':
    read_data()

    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # model = AutoModel.from_pretrained('bert-base-uncased').to(DEVICE)



