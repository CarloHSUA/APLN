import faiss
import os

class FaissWrapper():
    def __init__(self, filename) -> None:
        self.filename = filename
        self.faiss_db = None
        if os.path.isfile(filename):
            self.faiss_db = faiss.read_index(filename)

    def add_embeddings(self, x) -> None:
        '''Adds new embeddings auto calculating their indexes'''
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexFlatL2(x.shape[1])

        self.faiss_db.add(x)

    def add_embeddings_with_ids(self, x, ids):
        '''Adds data embeddings with their respective ids'''
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexIDMap(faiss.IndexFlatL2(x.shape[1]))

        self.faiss_db.add_with_ids(x=x, ids=ids)

    def save_to_disk(self) -> bool:
        if self.filename != "":
            faiss.write_index(self.faiss_db, self.filename)
            return True
        return False

    def query(self, query, k):
        return self.faiss_db.search(query, k=k)