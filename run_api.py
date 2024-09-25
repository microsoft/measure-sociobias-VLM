from tqdm import tqdm
import webuiapi

fname = 'all_biases_prompts.txt'
output_dir = './prompted_images/'
with open(fname, 'r') as f:
    prompts = [line.strip() for line in f.readlines()]

api = webuiapi.WebUIApi(host='127.0.0.1', port=7861)

for prompt in tqdm(prompts):
    sentence = 'a sketch of 1 faceless blue humanoid ' + prompt
    result = api.txt2img(
        prompt=sentence,
        negative_prompt='ugly, blurry, out of frame',
        seed=1618,
        cfg_scale=9.5,
        width=512,
        height=512,
        steps=24,
        styles=['sketch']
    )
    sentence = sentence if sentence.endswith('.') else sentence + '.'
    result.image.save(output_dir + sentence + 'png')
