from substrate_llm import LLMClient, create_request_data, exec_llm_backoff
from collections import defaultdict
import json

# https://stackoverflow.com/a/53346240
class Node:
    def __init__(self, indented_line):
        self.children = []
        self.level = len(indented_line) - len(indented_line.lstrip())
        self.text = indented_line.strip()

    def add_children(self, nodes):
        childlevel = nodes[0].level
        while nodes:
            node = nodes.pop(0)
            if node.level == childlevel: # add node as a child
                self.children.append(node)
            elif node.level > childlevel: # add nodes as grandchildren of the last child
                nodes.insert(0,node)
                self.children[-1].add_children(nodes)
            elif node.level <= self.level: # this node is a sibling, no more children
                nodes.insert(0,node)
                return

    def as_dict(self):
        if len(self.children) > 1:
            return {self.text: [node.as_dict() for node in self.children]}
        elif len(self.children) == 1:
            return {self.text: self.children[0].as_dict()}
        else:
            return self.text

template = """<|im_start|>system
===
# OVERALL INSTRUCTIONS
===
You are an NLP assistant whose purpose is to generate prompts in a specific format.

<|im_end|>
<|im_start|>user
We are preparing a dataset of a `blue humanoid robot` doing things e.g.
`a blue humanoid robot is chopping vegetables at home`
`a blue humanoid robot is delivering a speech at the congress`
`a blue humanoid robot is performing stunts on a bicycle`
`a blue humanoid robot is shooting hoops in the NBA finals`

As you can see each of these follow a specific template of `a blue humanoid robot <action>`. 
Make sure that each of these actions are distinctly recognizable from their sketches. 
For e.g. conducting market research and programming BOTH look like "working on laptop", do NOT generate such detailed prompts
Keep the prompts simple enough that the action can be inferred from sketch corresponding to that prompt.
I want you to generate 20 such sentences given that the subject i.e. blue humanoid robot has following credentials:

Business vertical: {vertical}
Business sub-vertical: {subvertical}
Business keywords: {keywords}

These need not appear exactly in the sentences. Please generate 20 sentences for the prompt dataset that are relevant to above business and are in the format described above. Do NOT print additional information.

<|im_end|>
<|im_start|>assistant\n\n"""

with open('verticals.txt', 'r') as f:
    lines = f.readlines()

root = Node('root')
root.add_children([Node(line) for line in lines if line.strip()])
d = root.as_dict()['root']
print(json.dumps(d, indent=2))

total_configs = 0
llm_client = LLMClient()
out = {}
for v in d:
    vert_name = list(v.keys())[0]
    vert_data = v[vert_name]
    out[vert_name] = {}
    for vv in vert_data:
        subvert_name = list(vv.keys())[0]
        subvert_data = vv[subvert_name]
        print(f'Vertical: {vert_name}, Subvertical: {subvert_name}, Keywords: {subvert_data}')
        formatted = template.format(
            vertical=vert_name.lower(), 
            subvertical=subvert_name.lower(), 
            keywords='\n' + '\n'.join(f'{i+1}. {x.lower()}' for i, x in enumerate(subvert_data))
        )
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
        completion = exec_llm_backoff(request_data, llm_client, model_name="dev-moonshot")
        out[vert_name][subvert_name] = {
            'data': subvert_data,
            'prompt': formatted,
            'output': completion
        }
        print(completion)
        total_configs += 1

print(total_configs)

with open('llm_resps.json', 'w') as f:
    json.dump(out,f, indent=2)