import torch
from lib.rag import RAG

if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rag = RAG(reload=False, device=DEVICE)
    rag.load_data()