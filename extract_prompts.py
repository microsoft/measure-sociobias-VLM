import pandas as pd
from glob import glob
from tqdm import tqdm
import re

files = '/home/t-assathe/InclusivityToolkit/inclusivity_toolkit/evaluators/EXTRINSIC/data_English/*/*.tsv'

sentences = []
for f in glob(files):
    df = pd.read_csv(f, sep='\t', header=None)
    sentences.extend(list(set(df[0])))
    sentences.extend(list(set(df[1])))
sentences = set(sentences)
#sentences = set(' '.join(x.split(' ')[2:]) for x in sentences)
#sentences = [re.sub('^person', '', x).strip() for x in sentences]
print('\n'.join(sentences))
print(len(sentences))

with open('all_biases_prompts.txt', 'w') as f:
    f.write('\n'.join(sentences))