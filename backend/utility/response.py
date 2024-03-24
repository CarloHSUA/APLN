import sys
sys.path.append('../..')
from lib.rag import RAG
import torch
import requests
import json


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# Load RAG model
rag = RAG(reload=False, device=DEVICE, verbose=1)
rag.load_data()
model = 'llama2:7b'


def __parse_response(response_text):
    # Dividir el texto en líneas
    return json.loads(response_text.strip().split('\n')[0]) # result_list


def __post_response(query: str):
    url = f"http://localhost:11434/api/generate"
    response = requests.post(url, json={'model': model, 'prompt': query, 'stream': False})
    return response.content.decode('utf-8')


def get_response(query: str, num_resposnes: int):
    I = rag.query(query, num_resposnes)[num_resposnes]
    story = rag.retrieve(I, columns=['story']).to_numpy()[0][0]
    context = f'''
    This is the context: "{story}",
    Answer the following question: {query}
    '''
    response = __post_response(context)
    return __parse_response(response)['response']
