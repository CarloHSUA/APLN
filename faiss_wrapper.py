import faiss
import os

class FaissWrapper():
    def __init__(self, filename) -> None:
        self.filename = filename
        self.faiss_db = None
        if os.path.isfile(filename):
            self.faiss_db = faiss.read_index(filename)

    def add_embeddings(self, corpus) -> None:
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexFlatL2(corpus.shape[1])

        self.faiss_db.add(corpus)

    def save_to_disk(self) -> bool:
        if self.filename != "":
            faiss.write_index(self.faiss_db, self.filename)
            return True
        return False

    def query(self, query, k):
        return self.faiss_db.search(query, k=k)