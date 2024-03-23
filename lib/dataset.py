import numpy as np
import pandas as pd
import os
from datasets import load_dataset

class CorpusReader:
    def __init__(self, source_filename, reload = True, verbose = 0) -> None:
        self.reload = reload
        self.verbose = verbose
        self.corpus_filename = source_filename
        self.corpus_dataset = None
        self.get_corpus_dataset()

    def load_corpus(self, columns):
        if self.verbose > 0:
            print("Downloading dataset")
        hf_dataset = load_dataset("glnmario/news-qa-summarization", split='train')
        if self.verbose > 0:
            print("Downloaded dataset")

        self.corpus_dataset = pd.DataFrame.from_dict(hf_dataset)
        # Creates column index
        self.corpus_dataset['index'] = np.arange(self.corpus_dataset.shape[0])
        self.corpus_dataset = self.corpus_dataset[columns]

    def get_corpus_dataset(self):
        if self.reload or not os.path.isfile(self.corpus_filename):
            self.load_corpus(['index', 'story', 'summary'])
            if self.verbose > 0:
                    print("Loaded documents from Hugging Face")
            self.save_corpus_to_disk()
        else:
            self.load_corpus_from_disk()
            print(f"Loaded documents from {self.corpus_filename}")
        
    def load_corpus_from_disk(self):
        self.corpus_dataset = pd.read_csv(self.corpus_filename)
        self.corpus_dataset['embeddings'] = pd.Series()

    def save_corpus_to_disk(self):
        self.corpus_dataset.to_csv(self.corpus_filename)
        if self.verbose > 0:
            print("Saved documents to disk")