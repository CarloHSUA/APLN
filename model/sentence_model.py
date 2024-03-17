from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import spacy  # Reconocimiento de entidades

# Instalamos el modelo en inglés de SpaCy
spacy.cli.download('en_core_web_sm')

class SencenceModel():
    def __init__(self, device) -> None:
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
        self.device = device
        spacy.cli.download('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_sm')


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def calculate_embedding(self, sentences):
        
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        self.model.eval()
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        return F.normalize(sentence_embeddings, p=2, dim=1)

