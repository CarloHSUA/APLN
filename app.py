import sys
import os
sys.path.append('../')
from model.sentence_model import SencenceModel
import torch
from lib.faiss_wrapper import FaissWrapper
from lib.preprocess_dataset import Dataset
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    model = SencenceModel(device = 'cpu')
    query = "Where were they being deployed to?"
    embeddings = np.expand_dims(np.array(model.calculate_embedding([query])[0]), axis=0)
    dataset = Dataset(reload = False, save_datasets = False, verbose = 1, generate_embeddings = False)
    index = FaissWrapper(f".{os.sep}Data{os.sep}cnn_news.index")
    print(f"Embeddings: {embeddings}")
    D, I =index.faiss_db.search(embeddings, k=5)
    print(I)
    print(dataset.sentence_dataset.iloc[I[0]])



if __name__ == '__main__':
    main()
