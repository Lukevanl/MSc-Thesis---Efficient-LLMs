import json
import os
import tiktoken
import numpy as np

print(os.listdir())

enc = tiktoken.get_encoding("gpt2")

for set_name in ['train', 'dev', 'test']:
    with open(f'./data/mednli/mli_{set_name}_v1.jsonl') as f:
        finetune_sentences = {'context': [], 'target': []}
        for line in f:
            data = json.loads(line)
            sentence_1 = data['sentence1']
            sentence_2 = data['sentence2']
            label = data['gold_label']
            context = f'Premise: {sentence_1} Hypothesis: {sentence_2} '
            label_to_int = {'entailment': 1, 'neutral': 2, 'contradiction' : 3}
            label = label_to_int[label]
            target = f'Target: the label for this hypothesis with respect to the premise is: {label}'
            finetune_sentences['context'].append(enc.encode_ordinary(context))
            finetune_sentences['target'].append(enc.encode_ordinary(target))
        with open(f"./data/mednli/{set_name}_gpt.json", "w") as outfile: 
            json.dump(finetune_sentences, outfile)
