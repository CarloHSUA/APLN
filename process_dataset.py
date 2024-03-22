import torch
from lib.rag import RAG

if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rag = RAG(reload=False, device=DEVICE, verbose=1)
    rag.load_data()
    D, I = rag.query('Who train in mock Afghan village before deployment to Afghanistan?', 4)
    print(rag.retrieve(I))