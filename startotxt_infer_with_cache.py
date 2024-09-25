from generators import *
from prompts import *
from PIL import Image
from tqdm import tqdm
from simple_parsing import ArgumentParser
from loguru import logger as lgr
from dataclasses import dataclass
import json

@dataclass
class Options:
    root_dir: str = './outputs/' # dir where you want to save the outputs
    prompts_path: str = './avg_bertscore.json' # Path to prompts
    images_path: str = '/home/t-ashsathe/BlobStorage/containers/absathe/MultiModalBias/final_generations' # Path to images
    model_name: str = 'gemini_vision' # model name: gemini, gpt4, llava, bakllava, codi
    task: str = 'img2txt' # task: txt2txt, img2txt, img2img, txt2img
    informed: bool = True # should you give info?
    direct: bool = True # should you prompt directly?
    kind: str = 'gender' # For img2txt only

def get_out_fname(args):
    return f'{args.root_dir}/{args.model_name}.{args.task}.informed={args.informed}.direct={args.direct}.kind={args.kind}.json'

@lgr.catch()
def main():
    parser = ArgumentParser(add_config_path_arg=True)
    parser.add_arguments(Options, dest='all')
    args = parser.parse_args().all

    out_fname = get_out_fname(args)
    prompt_template = ALL_PROMPTS[args.task][args.informed][args.direct]
    client, request_func = setup_model(args.model_name)

    with open(args.prompts_path) as f:
        data = json.load(f)
    
    if os.path.exists(out_fname):
        with open(out_fname, 'r') as f:
            out_data = json.load(f)
    else:
        out_data = []
    
    for i, item in enumerate(tqdm(data)):
        occ = item['gold_occupation'].replace('/', ' or ')
        action = list(item['predictions'].keys())[0]
        action = action.replace('A <subject> is ', '')
        if len(out_data) <= i:
            out_data.append({
                'gold_occupation': occ,
                'action': action,
                'responses': []
            })
        for j, subject in enumerate(['humanoid_robot', 'man', 'woman']):
            if len(out_data[i]['responses']) <= j:
                out_data[i]['responses'].append({
                    'arguments': [],
                    'response': ''
                })
            if len(out_data[i]['responses'][j]['response']) < 1:
                options_string = get_options_string(args.kind)
                prompt = prompt_template.format(occupation=occ, action=action, options_string=options_string)
                image_path = f'{args.images_path}/{subject}/{occ}.png'
                img = Image.open(image_path)
                resp = request_func(client, prompt, img)
                out_data[i]['responses'][j]['arguments'] = [prompt, image_path]
                out_data[i]['responses'][j]['response'] = resp
                with open(out_fname, 'w') as f:
                    json.dump(out_data, f, indent=2)
            print(out_data[i]['responses'][j])

if __name__ == '__main__':
    main()