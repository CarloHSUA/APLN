from nlgmetricverse import NLGMetricverse, load_metric
from rich import print
from lib.rag import RAG
import torch
import ast
import json, requests
import csv
from tqdm import tqdm

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
rag = RAG(device=DEVICE, reload=False, verbose=1)
rag.load_data()
csv_name = 'mistral.csv'

# If you specify more metrics, each of them will be applyied on your data (allowing for a fast prediction/efficiency comparison)
metrics = [
    load_metric("bertscore", compute_kwargs={"model_type": "microsoft/deberta-large-mnli"}),
    load_metric("bleu"),
    load_metric("rouge"),
    load_metric("meteor")
]
scorer = NLGMetricverse(metrics=metrics)

model = 'llama2:7b'

n_rows = len(rag.dataset.corpus_dataset)
sum_score = 0

def __parse_response(response_text):
    # Dividir el texto en lÃ­neas
    return json.loads(response_text.strip().split('\n')[0]) # result_list


def __post_response(query: str):
    url = f"http://localhost:11434/api/generate"
    response = requests.post(url, json={'model': model, 'prompt': query, 'stream': False})
    return response.content.decode('utf-8')

def write_csv(data):
    with open(csv_name, 'a', newline='') as csvfile:
        csvfile.write(','.join(data)+'\n')

for i in tqdm(range(n_rows), desc='Creating evaluation file'):
    row = rag.dataset.corpus_dataset.iloc[i]
    answers = ast.literal_eval(row['answers'])
    scores = []
    for idx, question in enumerate(ast.literal_eval(row['questions'])):
        D, I = rag.query(question, 1)
        context = rag.retrieve(I, columns=['story']).to_numpy()[0][0]
        context = context[:3800] if len(context) > 3800 else context
        context = f'''
        Make a short response. 
        You shouldn't make a reference to the question neither to the context.
        This is the context: "{context}",
        Answer the following question: {question}
        '''
        
        response = __post_response(context)
        predictions = __parse_response(response)['response']

        score = scorer.evaluate(predictions=[predictions], references=[answers[idx][0]])

        out = [score['bertscore']['score'],
               score['bertscore']['precision'][0],
               score['bertscore']['recall'][0],
               score['bleu']['score'],
               score['rouge']['rouge1'],
               score["meteor"]["score"]]
        
        scores.append(out)

    with open('results.csv', 'a', newline='') as f:
        csv.writer(f).writerows(scores)