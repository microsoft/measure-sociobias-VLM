from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

model_names = ['llava-1.5-7b-hf', 'bakLlava-v1-hf', 'gemini-pro-vision', 'gpt4v', 'codi']
model_mapping = {
    'llava-1.5-7b-hf': 'LLaVA',
    'bakLlava-v1-hf': 'BakLLaVA',
    'gemini-pro-vision': 'GeminiProVision',
    'gpt4v': 'GPT4V',
    'codi': 'CoDi'
}
model_order = list(model_mapping.values())

suffix = 'blind_direct'
#suffix = 'blind_indirect'
#suffix = 'informed_direct'
#suffix = 'informed_indirect'
with open(f'../jsons/profession_coarse_{suffix}.json') as f:
    data = json.load(f)

records = defaultdict(lambda : defaultdict(float))
for prof in data.keys():
    for model in data[prof].keys():
        records[prof][model] = (data[prof][model]['female'] - data[prof][model]['male']) / (data[prof][model]['total'])

df = pd.DataFrame(records).transpose()
for name in model_names:
    df[model_mapping[name]] = df[name]
df.drop(model_names, axis=1, inplace=True)
df = df[model_order]
df = df[model_order]
print(df)
print(df.min(axis=None), df.max(axis=None))

fig = plt.figure(figsize=(3, 8))
sns.heatmap(df, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.savefig(f'./heatmap_avggender_{suffix}.pdf', bbox_inches='tight')

plt.clf()
records = defaultdict(lambda : defaultdict(float))
for prof in data.keys():
    for model in data[prof].keys():
        records[prof][model] = data[prof][model]['neutrality']

df = pd.DataFrame(records).transpose()
for name in model_names:
    df[model_mapping[name]] = df[name]
df.drop(model_names, axis=1, inplace=True)
df = df[model_order]
print(df)
print(df.min(axis=None), df.max(axis=None))

fig = plt.figure(figsize=(3, 8))
sns.heatmap(df, annot=False, cmap='gist_gray', vmin=0, vmax=1, center=0.5)
plt.savefig(f'./heatmap_neutrality_{suffix}.pdf', bbox_inches='tight')