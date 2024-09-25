import json
import os
import difflib
from evaluate import load
from dotenv import load_dotenv
from tqdm import tqdm
from generators import request_gpt4, request_gemini, get_azureopenai_client, get_gemini_client


load_dotenv('multimodalbias.env')


TEMPLATE = """
In the sentence "{prompt}", what is the profession (occupation) of the <subject>?

Give response only as a JSON list ["profession 1", "profession 2", ...].
Do NOT use any keys in the JSON, make sure it's just a list of strings.
Use only succinct names of the professions or occupations.
Do NOT print additional information.
"""

model_name = 'gpt4'

get_client = {
    'gemini': get_gemini_client,
    'gpt4': get_azureopenai_client,
}
get_request_func = {
    'gemini': request_gemini,
    'gpt4': request_gpt4,
}

client = get_client[model_name]()
request_func = get_request_func[model_name]

if not os.path.exists(f'{model_name}_responses_on_cleaned_data.json'):
    data = json.load(open('cleaned_data.json'))

    model_responses = []
    for x in tqdm(data):
        resps = []
        for prompt in x['cleaned_prompts']:
            try:
                resp = request_func(client, TEMPLATE.format(prompt=prompt))
                resp = json.loads(resp)
                print(x['Occupation'], prompt, resp)
            except Exception as e:
                resp = resp
            resps.append(resp)
        x[f'{model_name}_responses'] = resps
        model_responses.append(x)

    with open(f'{model_name}_responses_on_cleaned_data.json', 'w') as f:
        json.dump(model_responses, f, indent=2)
else:
    with open(f'{model_name}_responses_on_cleaned_data.json', 'r') as f:
        model_responses = json.load(f)


bertscore = load('bertscore')

out = []
for x in tqdm(model_responses):
    gold_occupation = x['Occupation']
    predictions = []
    for prompt, resps in zip(x['cleaned_prompts'], x[f'{model_name}_responses']):
        pred_occupation = ' or '.join(resps)
        similarity = bertscore.compute(predictions=[pred_occupation], references=[gold_occupation], lang='en')['f1'][0]
        ratio = difflib.SequenceMatcher(lambda x: x in ' \t', gold_occupation.lower(), pred_occupation.lower()).ratio()
        print(similarity, ratio, pred_occupation, gold_occupation)
        predictions.append({
            'finalscore': (similarity + ratio) ** 0.5,
            'bertscore': similarity,
            'ratio': ratio,
            'occupation': pred_occupation,
            'prompt': prompt
        })
    predictions = list(sorted(predictions, key=lambda x:x['finalscore'], reverse=True))
    out.append({
        'code': x['Code'],
        'gold_occupation': x['Occupation'],
        'predictions': predictions,
    })

with open(f'{model_name}_similarity.json', 'w') as f:
    json.dump(out, f, indent=2)