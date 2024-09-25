import json
with open('./gemini-pro-vision.json') as f:
    data = json.load(f)

with open('./fnames.txt', 'w') as f:
    f.write('\n'.join(data.keys()))