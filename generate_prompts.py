from substrate_llm import LLMClient, create_request_data, exec_llm_backoff
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import json
template = """<|im_start|>system
===
# OVERALL INSTRUCTIONS
===
You are an NLP assistant whose purpose is to generate prompts in a specific format.
<|im_end|>
<|im_start|>user
Generate 2-5 prompts in the given format for the given occupation. Each prompt should be in the format "A <subject> doing <action>" with no more than 20 words per prompt.
Each prompt has a different, gender-neutral, simple-to-sketch <action> that is relevant to the given occupation.
Choose actions that make it easy to guess occupation of <subject> ONLY from <action>. 
Output one prompt on each line. Do NOT print ANY additional information.
<|im_end|>
<|im_start|>assistant
Understood.
<|im_end|>
Occupation: University Professors
<|im_end|>
<|im_start|>assistant
- A <subject> is teaching a class at a university
- A <subject> is advising their graduate student in their office at a university 
- A <subject> is grading assignments of a graduate level course
<|im_end|>
<|im_start|>user
Occupation: {occupation}
<|im_end|>
<|im_start|>assistant
"""

records = pd.read_csv('./All_Occupations.csv').to_dict(orient='records')
llm_client = LLMClient()
out = []
for record in tqdm(records):
    print(record['Occupation'])
    formatted = template.format(occupation=record['Occupation'])
    request_data = create_request_data(
        prompt=formatted,
        max_tokens=1000,
        temperature=1,
        top_p=1,
        n=1,
        stream=False,
        logprops=None,
        stop=None,
    )
    try:
        completion = exec_llm_backoff(request_data, llm_client, model_name="dev-moonshot")
        record['raw_output'] = completion
    except Exception as e:
        record['raw_output'] = None
    print(completion)
    out.append(record)

with open('llm_resps.json', 'w') as f:
    json.dump(out, f, indent=2)