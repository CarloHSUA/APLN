import torch
from lib.rag import RAG

if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rag = RAG(reload=False, device=DEVICE, verbose=1)
    rag.load_data()
    query = "What is the number of pink used vehicles listed?"
    D, I = rag.query(query, 10)
    print(rag.retrieve(I))