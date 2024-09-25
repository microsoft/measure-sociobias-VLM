import json

with open('./final_fnames.txt') as f:
    prompts = [line.strip().lower().replace('a blue humanoid robot', '').replace('.png', '') for line in f.readlines()]
with open('./jsons/llm_resps.json') as f:
    llm_resps = json.load(f)

prompt_to_occ = {}
for prompt in prompts:
    occ = 'Corporate.Technology'
    found = False
    for k1, v1 in llm_resps.items():
        for k2, v2 in v1.items():
            if prompt in v2['output'].lower():
                occ = f'{k1}.{k2}'
                found = True
                break
        if found:
            break
    prompt_to_occ[prompt] = occ

with open('./jsons/mapping.json', 'w') as f:
    json.dump(prompt_to_occ, f, indent=2)