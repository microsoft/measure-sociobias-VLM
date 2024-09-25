from collections import defaultdict
import pandas as pd
import json

models = [
    'llava-1.5-7b-hf',
    'bakLlava-v1-hf',
    'gemini-pro-vision',
    'gpt4v'
]

keys = [l.strip() for l in open('fnames.txt').readlines()]

def process_models(model_names, prefix='', suffix=''):
    datapoints = defaultdict(lambda : defaultdict(str))
    for model_name in models:
        with open(prefix + model_name + suffix + '.json') as f:
            data = json.load(f)
        for k, v in data.items():
            datapoints[k.lower()][model_name] = v
    datapoints = [{'fname': k, **v} for k, v in datapoints.items()]
    df = pd.DataFrame(datapoints)
    return df

all_dfs = {}
for prefix in ['', 'casting_']:
    for suffix in ['', '_informed']:
        all_dfs[('results_' + prefix + suffix).replace('_', '')] = process_models(models)

with pd.ExcelWriter('all_results.xlsx') as writer:
    for k, v in all_dfs.items():
        v.to_excel(writer, sheet_name=k)