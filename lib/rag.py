from lib.dataset import CorpusReader
from lib.vector_database import FaissWrapper
from model.sentence_model import SencenceModel
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class RAG:
    def __init__(self, device, model = None, reload = False, verbose = 0) -> None:
        self.device = device
        self.model = SencenceModel(device) if model is None else model
        self.reload = reload
        self.verbose = verbose
        self.dataset_filename = f'.{os.sep}data{os.sep}source_dataset.csv'
        self.faiss_filename = f'.{os.sep}data{os.sep}cnn_news.index'
        self.dataset = None
        self.faiss = None

    def load_data(self):
        self.dataset = CorpusReader(self.dataset_filename, reload = self.reload, verbose = self.verbose)
        self.faiss = FaissWrapper(self.faiss_filename, verbose = self.verbose)
        if self.faiss.faiss_db is None:
            self.__process_documents__()

    def __process_documents__(self):
        batch_size = 64
        with tqdm(total=len(self.dataset.corpus_dataset), desc="Processing documents") as pbar:
            for i in range(0, len(self.dataset.corpus_dataset), batch_size):
                batch = self.dataset.corpus_dataset.iloc[i : i + batch_size]
                embeddings = self.model.calculate_embedding(list(batch['summary']))
                self.faiss.add_embeddings_with_ids(embeddings.cpu().numpy(), batch['index'].to_numpy())
                pbar.update(len(batch))

        self.faiss.save_to_disk()

    def query(self, query, k = 5):
        if query == '':
            return []

        embedding_query = self.model.calculate_embedding([query]).cpu().numpy()
        D, I = self.faiss.query(embedding_query, k)
        return D, I
    
    def retrieve(self, ids, columns = ['story','summary']):
        return self.dataset.iloc[ids][columns]
