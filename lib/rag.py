import os
from lib.dataset import CorpusReader
from lib.vector_database import FaissWrapper
from model.sentence_model import SencenceModel
from tqdm import tqdm

class RAG:
    def __init__(self, device, model = None, reload = False, verbose = 0) -> None:
        '''Creates a new object of type RAG. This object wraps most of the basic functionality used in a RAG architecture.

        :param str device: Device where most of the code will be run on.
        :param obj model: Model used to convert text into embeddings.
        :param bool reload: Indicates if the files are created again.
        :param int verbose: Level of verbose.
        '''
        self.device = device
        self.model = SencenceModel(device) if model is None else model
        self.reload = reload
        self.verbose = verbose
        self.dataset_filename = f'.{os.sep}data{os.sep}source_dataset.csv'
        self.faiss_filename = f'.{os.sep}data{os.sep}cnn_news.index'
        self.dataset = None
        self.faiss = None

    def load_data(self):
        '''Loads into memory the two files used in the step of retrieval information. If they exist they will be loaded and 
        if they don't or reload is selected, they will be generated.
        '''
        self.dataset = CorpusReader(self.dataset_filename, reload = self.reload, verbose = self.verbose)
        self.faiss = FaissWrapper(self.faiss_filename, reload = self.reload, verbose = self.verbose)
        if self.faiss.faiss_db is None:
            self.__process_documents__()

    def __process_documents__(self):
        '''Processes all documents in the dataset and generate their embeddings to create the faiss index.'''
        batch_size = 64
        with tqdm(total=len(self.dataset.corpus_dataset), desc="Processing documents") as pbar:
            for i in range(0, len(self.dataset.corpus_dataset), batch_size):
                # Get batch from pandas dataframe
                batch = self.dataset.corpus_dataset.iloc[i : i + batch_size]
                # Generates the embeddings for the whole batch
                embeddings = self.model.calculate_embedding(list(batch['summary']))
                # Adds these embeddings to faiss index
                self.faiss.add_embeddings_with_ids(embeddings.cpu().numpy(), batch['index'].to_numpy())
                pbar.update(len(batch))

        self.faiss.save_to_disk()

    def query(self, query, k = 5):
        '''Query against all processed files and return the k most relevant document indexs. 

        :param str query: Query in natural language.
        :param int k: Search for the k-nearest.
        :return List<float> D: List of distances between the query and the retrieved documents.
        :return List<long int> I: List of indixes of the k-nearest documents.
        '''
        if query == '':
            return []

        # Calculates the embedding of the query and search the k closest in faiss index
        embedding_query = self.model.calculate_embedding([query]).cpu().numpy()
        D, I = self.faiss.query(embedding_query, k)
        return D, I[0]
    
    def retrieve(self, ids, columns = ['story','summary']):
        '''Retrieves the selected columns from the documents using the given indexs. 

        :param list<long int> ids: Ids of the documents.
        :param list<str> columns: Which column are retrieved in the dataframe.
        '''
        return self.dataset.corpus_dataset.iloc[ids][columns]
