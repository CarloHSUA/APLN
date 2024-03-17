import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset

hf_dataset = load_dataset("glnmario/news-qa-summarization", split='train')


corpus_dataset = pd.DataFrame.from_dict(hf_dataset)
# Creates column index
corpus_dataset['index'] = np.arange(corpus_dataset.shape[0])

# Reorder columns
corpus_dataset = corpus_dataset[['index', 'story', 'questions', 'answers', 'summary']]
# Saves the dataset from Hugging Face
corpus_dataset.to_csv('source_dataset.csv')

# We download English model for Spacy
spacy.cli.download('en_core_web_sm')

# Loads the English model
nlp = spacy.load('en_core_web_sm')

sentence_list = []

def split_into_sentences(row):
    document = nlp(row['story'])
    for sentence in list(document.sents):
        new_row = {}
        new_row['doc_index'] = row['index']
        new_row['index'] = len(sentence_list) + 1
        new_row['sentence'] = sentence
        new_row['embeddings'] = '' # TODO
        sentence_list.append(new_row)

corpus_dataset.apply(split_into_sentences, axis=1)


sentence_dataset = pd.DataFrame(sentence_list, columns=['doc_index', 'index', 'sentence', 'embeddings'])
sentence_dataset.to_csv('sentence_dataset.csv')