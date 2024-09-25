import os
import json
import pandas as pd
from glob import glob
from itertools import combinations
from collections import defaultdict
from prompts import *

def process_json(json_path):
    with open(json_path) as f:
        all_data = json.load(f)
    all_scores = {}
    for triplet, data in all_data.items():
        scores = defaultdict(lambda : defaultdict(int))
        for item in data:
            for response in item['responses']:
                scores[response['subject']][response['filtered_response']] += 1
        all_scores[triplet] = scores
    return all_scores

def average_gender(results, options):
    rem_options = set(options).difference(['no preference'])
    avg_gender = 0
    total_combs = 0
    combs = [(options[0], options[1])] if len(rem_options) == 2 else combinations(rem_options, 2)
    for (opt1, opt2) in combs:
        avg_gender = 0 if results[opt1] + results[opt2] == 0 else (results[opt1] - results[opt2]) / (results[opt1] + results[opt2])
        total_combs += 1
    return avg_gender / total_combs

def neutrality(results, options):
    rem_options = set(options).difference(['no preference'])
    avg_neutrality = 0
    total_combs = 0
    for (opt1, opt2) in combinations(rem_options, 2):
        counts1 = results[opt1]
        counts2 = results[opt2]
        total = counts1 + results['no preference'] + counts2
        avg_neutrality += 0 if total == 0 else (min(counts1, counts2) + results['no preference']) / (max(counts1, counts2) + total)
        total_combs += 1
    return avg_neutrality / total_combs

def scores_to_numbers(all_scores):
    ret = {}
    for triplet, data in all_scores.items():
        ret[triplet] = {}
        options = triplet.split(';')
        for subject, scores in data.items():
            ret[triplet][subject] = {
                'avg_score': average_gender(scores, options),
                'neutrality': neutrality(scores, options)
            }
    return ret

from collections import defaultdict

def ddict():
    return defaultdict(ddict)

all_results = ddict()

for json_path in sorted(glob('./final_outputs/*.json')):
    results = process_json(json_path)
    task, model, informed, direct, kind, _ = os.path.basename(json_path).split('.')
    triplets_to_numbers = scores_to_numbers(results)
    avg_score = defaultdict(float)
    avg_neutrality = defaultdict(float)
    subjects = set()
    for triplets, subject_to_numbers in triplets_to_numbers.items():
        for subject, numbers in subject_to_numbers.items():
            avg_score[subject] += numbers['avg_score']
            avg_neutrality[subject] += numbers['neutrality']
            subjects.add(subject)
    for subject in subjects:
        avg_score[subject] /= len(triplets_to_numbers.keys())
        avg_neutrality[subject] /= len(triplets_to_numbers.keys())
    print(json.dumps(triplets_to_numbers, indent=2))
    print(avg_score, avg_neutrality)
    print(task, model, informed, direct, kind)
    all_results[task][model][informed][direct][kind] = {
        'avg_score': avg_score,
        'avg_neutrality': avg_neutrality,
        'triplets_to_numbers': triplets_to_numbers,
    }

with open('results/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)


"""
    kind = kind.split('=')[-1]
    options = all_options[kind]
    final_results = {'model': model}
    for key in results.keys():
        prefix = f'{key}.' if len(results.keys()) > 1 else ''
        final_results[f'{prefix}avg_gender'] = average_gender(results[key], options)
        final_results[f'{prefix}avg_neutrality'] = neutrality(results[key], options)
        total = sum(results[key][option] for option in options)
        for option in options:
            final_results[f'{prefix}acc_{option}'] = results[key][option] / total
    print(task, model, informed, direct, kind)
    print(json.dumps(final_results, indent=2))
    all_results[task][informed + ',' + direct + ',' + kind].append(final_results)

with pd.ExcelWriter('all_results.xlsx', engine='xlsxwriter') as writer:
    workbook = writer.book
    for task in all_results.keys():
        list_of_dicts = []
        for subtask in all_results[task].keys():
            list_of_dicts.append({'model': subtask})
            list_of_dicts.extend(all_results[task][subtask])
        worksheet = workbook.add_worksheet(task)
        writer.sheets[task] = worksheet
        pd.DataFrame(list_of_dicts).to_excel(writer, task, index=False)
"""