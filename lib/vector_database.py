import faiss
import os

class FaissWrapper():
    def __init__(self, filename, reload = False, verbose = 0) -> None:
        '''Creates a new object of type FaissWrapper. This object wraps basic functionality to read/write a faiss index.

        :param str filename: Name of the file where the index is stored.
        :param bool reload: Indicates if the files are created again.
        :param int verbose: Level of verbose.
        '''
        self.filename = filename
        self.verbose = verbose
        self.faiss_db = None
        self.reload = reload
        if not self.reload and os.path.isfile(filename):
            self.faiss_db = faiss.read_index(filename)
            if self.verbose > 0:
                print(f"Loaded Faiss index from {filename}")
        
    def add_embeddings(self, x) -> None:
        '''Adds new embeddings auto calculating their indexes.

        :param list<list<float>> x: List of embeddings that will be stored into faiss index.
        '''
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexFlatL2(x.shape[1])

        self.faiss_db.add(x)

    def add_embeddings_with_ids(self, x, ids):
        '''Adds data embeddings with their respective ids.

        :param list<list<float>> x: List of embeddings that will be stored into faiss index.
        :param list<long int> ids: Indexs of the documents related to the embeddings in the same order.
        '''
        if self.faiss_db is None:
            self.faiss_db = faiss.IndexIDMap(faiss.IndexFlatL2(x.shape[1]))

        # Adds the embeddings and the correspondant ids
        self.faiss_db.add_with_ids(x=x, ids=ids)

    def save_to_disk(self) -> bool:
        '''Saves the faiss index to disk'''
        if self.filename != "":
            faiss.write_index(self.faiss_db, self.filename)
            if self.verbose > 0:
                print(f"Wrote index to {self.filename}")
            return True
        return False

    def query(self, query, k):
        '''Query against all processed files and return the k most relevant document indexs. 

        :param str query: Query in natural language.
        :param int k: Search for the k-nearest.
        :return List<float> D: List of distances between the query and the retrieved documents.
        :return List<long int> I: List of indixes of the k-nearest documents.
        '''
        return self.faiss_db.search(query, k=k)