import numpy as np
import os
import pandas as pd
import spacy
from tqdm import tqdm
from datasets import load_dataset
from model.sentence_model import SencenceModel
import torch
from faiss_wrapper import FaissWrapper

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("DEVICE: ", DEVICE)

class Dataset:
    def __init__(self, reload = True, save_datasets = True, verbose = 0, generate_embeddings = True) -> None:
        self.reload = reload
        self.save_datasets = save_datasets
        self.verbose = verbose
        self.corpus_filename = 'source_dataset.csv'
        self.sentences_filename = 'sentence_dataset.csv'
        self.corpus_dataset = None
        self.sentence_dataset = None
        self.get_corpus_dataset()
        self.get_sentences_dataset(generate_embeddings)

    def load_corpus(self, columns):
        hf_dataset = load_dataset("glnmario/news-qa-summarization", split='train')
        if self.verbose > 0:
            print("Downloaded dataset")
        self.corpus_dataset = pd.DataFrame.from_dict(hf_dataset)

        # Creates column index
        self.corpus_dataset['index'] = np.arange(self.corpus_dataset.shape[0])
        # Reorder columns
        self.corpus_dataset = self.corpus_dataset[columns]

    def get_corpus_dataset(self):
        # Loads corpus dataset
        if self.reload or not os.path.isfile(self.corpus_filename):
            self.load_corpus(['index', 'story'])
            if self.verbose > 0:
                    print("Loaded documents from Hugging Face")
            if self.save_datasets:
                self.save_corpus_to_disk()
                if self.verbose > 0:
                    print("Saved documents to disk")
        else:
            self.load_corpus_from_disk()
            print(f"Loaded documents from {self.corpus_filename}")

    def get_sentences_dataset(self, generate_embeddings = True):
        # Loads sentence dataset
        if self.reload or not os.path.isfile(self.sentences_filename):
            self.process_sentences(generate_embeddings = generate_embeddings)
            if self.save_datasets:
                self.save_sentences_to_disk()
                if self.verbose > 0:
                    print("Saved sentences to disk")
        else:
            self.load_sentences_from_disk()
            print(f"Loaded documents from {self.sentences_filename}")
        
    def load_corpus_from_disk(self):
        self.corpus_dataset = pd.read_csv(self.corpus_filename)

    def save_corpus_to_disk(self):
        self.corpus_dataset.to_csv(self.corpus_filename)

    def load_sentences_from_disk(self):
        self.sentence_dataset = pd.read_csv(self.sentences_filename)
        self.sentence_dataset['embeddings'] = pd.Series()

    def save_sentences_to_disk(self):
        self.sentence_dataset[['doc_index', 'index', 'sentence']].to_csv(self.sentences_filename)

    def process_sentences(self, generate_embeddings = True):
        if not spacy.util.is_package("en_core_web_sm"):
            # We download English model for Spacy
            spacy.cli.download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

        sentence_list = []
        with tqdm(total=len(self.corpus_dataset), desc="Processing documents to sentences") as pbar:
            if generate_embeddings:
                model = SencenceModel(device = DEVICE)

            def split_into_sentences(row):
                document = nlp(row['story'])
                if generate_embeddings:
                    embeddings = model.calculate_embedding(list(map(lambda e: str(e), document.sents)))

                for sentence_idx, sentence in enumerate(list(document.sents)):
                    new_row = {}
                    new_row['doc_index'] = row['index']
                    new_row['sentence'] = sentence
                    new_row['embeddings'] = None if not generate_embeddings else embeddings[sentence_idx].cpu().numpy()
                    sentence_list.append(new_row)
                pbar.update(1)

            self.corpus_dataset.apply(split_into_sentences, axis=1)

        pbar.close()

        self.sentence_dataset = pd.DataFrame(sentence_list, columns=['doc_index', 'index', 'sentence', 'embeddings'])
        self.sentence_dataset['index'] = np.arange(self.sentence_dataset.shape[0])

    def generate_embeddings(self):
        if self.sentence_dataset['embeddings'].isnull().any(axis=0):
            model = SencenceModel(device = DEVICE)

            with tqdm(total=len(self.sentence_dataset), desc="Calculating embeddings") as pbar:
                def calculate_embedding(row):
                    embeddings = model.calculate_embedding(list(row['sentence']))
                    row['embeddings'] = embeddings[0].cpu()
                    pbar.update(1)

                self.sentence_dataset.apply(calculate_embedding, axis=1)

            pbar.close()

if __name__ == "__main__":
    dataset = Dataset(reload = False, save_datasets = False, verbose = 1, generate_embeddings = False)
    index = FaissWrapper("cnn_news.index")
    if index.faiss_db is None:
        dataset.generate_embeddings()
        index.add_embeddings_with_ids(np.vstack(dataset.sentence_dataset['embeddings']), dataset.sentence_dataset['index'])
        index.save_to_disk()