from nlgmetricverse import NLGMetricverse, load_metric
from rich import print

# If you specify more metrics, each of them will be applyied on your data (allowing for a fast prediction/efficiency comparison)
metrics = [
    load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
    load_metric("bleu", resulting_name="bleu_2", compute_kwargs={"max_order": 2}),
    load_metric("bertscore", resulting_name="bertscore_1", compute_kwargs={"model_type": "microsoft/deberta-large-mnli"}),
    load_metric("rouge")
    ]
scorer = NLGMetricverse(metrics=metrics)

predictions = ["The rat is huge"]
references = [
    ["Evaluating artificial text is not difficult", "Evaluating artificial text is simple"]
]

score = scorer.evaluate(predictions=predictions, references=references)

print(score)
