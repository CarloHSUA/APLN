import numpy as np
import os
import pandas as pd
from datasets import load_dataset

class CorpusReader:
    def __init__(self, source_filename, reload = True, verbose = 0) -> None:
        '''Creates a new object of type CorpusReader. This object wraps the reading and saving of the dataset used in the application.

        :param str source_filename: Name of the file where the dataset is stored.
        :param bool reload: Indicates if the files are created again.
        :param int verbose: Level of verbose.
        '''
        self.reload = reload
        self.verbose = verbose
        self.corpus_filename = source_filename
        self.corpus_dataset = None
        self.get_corpus_dataset()

    def load_corpus(self, columns):
        '''Reads the dataset from Hugging Face and stores the given columns.

        :param list<str> columns: List of columns that will be read and saved.
        '''
        if self.verbose > 0:
            print("Downloading dataset")
        hf_dataset = load_dataset("glnmario/news-qa-summarization", split='train')
        if self.verbose > 0:
            print("Downloaded dataset")

        # Converts the dataset from Hugging Face into pandas dataframe
        self.corpus_dataset = pd.DataFrame.from_dict(hf_dataset)
        # Creates column index
        self.corpus_dataset['index'] = np.arange(self.corpus_dataset.shape[0])
        # Selects the given columns
        self.corpus_dataset = self.corpus_dataset[columns]

    def get_corpus_dataset(self):
        '''If the file exists and no reload is request, it loads the dataset from local file. Otherwise, creates it anew.'''
        if self.reload or not os.path.isfile(self.corpus_filename):
            self.load_corpus(['index', 'story', 'summary'])
            if self.verbose > 0:
                    print("Loaded documents from Hugging Face")
            self.save_corpus_to_disk()
        else:
            self.load_corpus_from_disk()
            print(f"Loaded documents from {self.corpus_filename}")
        
    def load_corpus_from_disk(self):
        '''Loads the dataset from disk and loads it into memory.'''
        self.corpus_dataset = pd.read_csv(self.corpus_filename)
        self.corpus_dataset['embeddings'] = pd.Series()

    def save_corpus_to_disk(self):
        '''Saves the dataset to disk.'''
        self.corpus_dataset.to_csv(self.corpus_filename)
        if self.verbose > 0:
            print("Saved documents to disk")