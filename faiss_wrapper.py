import faiss
import os

class FaissWrapper():
    def __init__(self, filename) -> None:
        self.filename = filename
        self.faiss_db = None
        if os.path.isfile(filename):
            self.faiss_db = faiss.read_index(filename)

    def add_embeddings(self, corpus_pd) -> None:
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexFlatL2(len(corpus_pd['embeddings'][0]))

        self.faiss_db.add_with_ids(corpus_pd['embeddings'], xids=corpus_pd['index'])

    def save_to_disk(self) -> bool:
        if self.filename != "":
            faiss.write_index(self.faiss_db, self.filename)
            return True
        return False

    def query(self, query, k):
        return self.faiss_db.search(query, k=k)