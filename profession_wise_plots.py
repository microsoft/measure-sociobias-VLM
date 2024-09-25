import os
import json
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy
from itertools import combinations
from collections import defaultdict
from prompts import *
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'
plt.rcParams["figure.figsize"] = (6, 12)
plt.style.use('seaborn-v0_8-colorblind')

with open('./avg_bertscore.json') as f:
    prof_to_id = {datum['gold_occupation'].replace('/', ' or '): datum['code'].split('-')[0] for datum in json.load(f)}

def ddict():
    return defaultdict(ddict)

def process_json(json_path):
    with open(json_path) as f:
        all_data = json.load(f)
    all_scores = ddict()
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for triplet, data in all_data.items():
        for item in data:
            for response in item['responses']:
                code = prof_to_id[item['gold_occupation']]
                subject = response['subject']
                response = response['filtered_response']
                print(code, subject, response, triplet)
                scores[code][triplet][response] += 1
        all_scores[subject] = scores
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

# https://www.bls.gov/oes/current/oes_stru.htm
profession_to_name = {
	'11':'Management ', #Opccupations',
	'13':'Business and Financial Operations ', #Opccupations',
	'15':'Computer and Mathematical ', #Opccupations',
	'17':'Architecture and Engineering ', #Opccupations',
	'19':'Life, Physical, and Social Science ', #Opccupations',
	'21':'Community and Social Service ', #Opccupations',
	'23':'Legal ', #Opccupations',
	'25':'Educational Instruction and Library ', #Opccupations',
	'27':'Arts, Design, Entertainment, Sports, and Media ', #Opccupations',
	'29':'Healthcare Practitioners and Technical ', #Opccupations',
	'31':'Healthcare Support ', #Opccupations',
	'33':'Protective Service ', #Opccupations',
	'35':'Food Preparation and Serving Related ', #Opccupations',
	'37':'Building and Grounds Cleaning and Maintenance ', #Opccupations',
	'39':'Personal Care and Service ', #Opccupations',
	'41':'Sales and Related ', #Opccupations',
	'43':'Office and Administrative Support ', #Opccupations',
	'45':'Farming, Fishing, and Forestry ', #Opccupations',
	'47':'Construction and Extraction ', #Opccupations',
	'49':'Installation, Maintenance, and Repair ', #Opccupations',
	'51':'Production ', #Opccupations',
	'53':'Transportation and Material Moving ', #Opccupations',
}

def get_dominant(triplets_and_scores):
    from collections import Counter
    a = Counter()
    for triplet, scores in triplets_and_scores.items():
        for option, count in scores.items():
            a[option] += count
    return a.most_common(1)[0][0]

def scores_to_numbers(all_results):
    ret = {}
    for subject in all_results.keys():
        ret[subject] = {}
        for occ_id, triplets_and_scores in all_results[subject].items():
            if occ_id not in profession_to_name.keys():
                continue
            occ_name = profession_to_name[occ_id]
            avg_score = 0
            avg_neutrality = 0
            count = 0
            for triplet, scores in triplets_and_scores.items():
                avg_score += average_gender(scores, triplet.split(';'))
                avg_neutrality += neutrality(scores, triplet.split(';'))
                count += 1
            avg_score /= count
            avg_neutrality /= count
            ret[subject][occ_name] = {
                'avg_score': avg_score,
                'avg_neutrality': avg_neutrality,
                'most_dominant': get_dominant(triplets_and_scores),
            }
    return ret

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    #ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


friendly_names = {
    'llava': 'LLaVA',
    'vipllava': 'ViPLLaVA',
    'gemini_vision': 'GeminiProVision',
    'gpt4v': 'GPT4V',
    'codi': 'CoDi',
}

models_list = list(friendly_names.values())

informed = True
direct = True
kind = 'gender'

sorted_orders = {
    'gender': ['male', 'female', 'no preference'],
    'casting_us': ['Brad Pitt', 'Angelina Jolie', 'no preference'],
    'race': ['Caucasian', 'Asian', 'African American', 'no preference'],
    'race_indirect_1': ['Johnny Depp', 'Anil Kapoor', 'Djimon Hounsou', 'no preference'],
    'age': ['under 18 years', '18-44 years', '45-64 years', 'over 65 years', 'no preference'],
    'age_indirect_1': ['Iain Armitage', 'Noah Schnapp', 'James Franco', 'Robert Duvall', 'no preference'],
}

for informed in [True, False]:
    for direct in [True, False]:
        kind = 'age' if direct else 'age_indirect_1'
        options = sorted_orders[kind]
        occ_to_metrics = ddict()
        for json_path in sorted(glob(f'./final_outputs/img2txt*informed={informed}.direct={direct}.kind={kind}.json')):
            results = process_json(json_path)
            _, model, _, _, _, _ = os.path.basename(json_path).split('.')
            for occ_name, record in scores_to_numbers(results)['humanoid_robot'].items():
                occ_to_metrics['avg_score'][occ_name][friendly_names[model]] = record['avg_score']
                occ_to_metrics['avg_neutrality'][occ_name][friendly_names[model]] = record['avg_neutrality']
                occ_to_metrics['most_dominant'][occ_name][friendly_names[model]] = options.index(record['most_dominant'])
        print(informed, direct, kind)
        print(pd.DataFrame(occ_to_metrics['avg_score']))
        avg_score = pd.DataFrame(occ_to_metrics['avg_score']).T[models_list]
        avg_neutrality = pd.DataFrame(occ_to_metrics['avg_neutrality']).T[models_list]
        most_dominant = pd.DataFrame(occ_to_metrics['most_dominant']).T[models_list]

        plt.clf()

        im, cbar = heatmap(
            avg_score.to_numpy(), 
            row_labels = avg_score.index,
            col_labels = avg_score.columns,
            ax = plt.gca(),
            cmap = 'bwr',
        )
        texts = annotate_heatmap(im, valfmt='{x:.2f}')

        plt.tight_layout()

        plt.savefig(f'plots/img2txt.informed={informed}.direct={direct}.age.avg_score.pdf')

        plt.clf()

        im, cbar = heatmap(
            avg_neutrality.to_numpy(), 
            row_labels = avg_neutrality.index,
            col_labels = avg_neutrality.columns,
            ax = plt.gca(),
            cmap = 'Purples',
        )
        texts = annotate_heatmap(im, valfmt='{x:.2f}')

        plt.tight_layout()

        plt.savefig(f'plots/img2txt.informed={informed}.direct={direct}.age.avg_neutrality.pdf')

        plt.clf()

        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: options[most_dominant.to_numpy()[x][pos]])
        print(most_dominant.to_numpy().min())
        print(most_dominant.to_numpy().max())
        im, cbar = heatmap(
            most_dominant.to_numpy(), 
            row_labels = most_dominant.index,
            col_labels = most_dominant.columns,
            ax = plt.gca(),
            cmap = matplotlib.colormaps['Pastel1'].resampled(len(options)),
            cbar_kw = dict(
                ticks = np.arange(len(options)),
            )
        )
        cbar.ax.set_yticklabels(options)
        #texts = annotate_heatmap(im, valfmt='{x:.2f}')

        plt.tight_layout()

        plt.savefig(f'plots/img2txt.informed={informed}.direct={direct}.age.most_dominant.pdf')