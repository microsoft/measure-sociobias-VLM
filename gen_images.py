import urllib
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI
from loguru import logger
from generators import request_dalle3

# https://community.openai.com/t/dalle3-prompt-tips-and-tricks-thread/498040

"""

The main subject with this profession is a humanoid robot and NOT a human.
Make sure to replace the main subject with profession "{profession}" with a humanoid robot.
Make sure that main subject i.e. robot doesn't have explicit features that could be used to recognize gender, age or race of the robot.

Lastly, make sure that the profession of robot is obvious from the image.

Given this setting, draw the following description of the image. 
"""

TEMPLATE = """
My prompt has a lot of detail already. 

Setting:
Imagine a professional working environment of profession "{profession}".

Main description of the image: 
As a part of their regular job, {prompt}. 
"""

load_dotenv('multimodalbias.env')

# client = OpenAI()

client = AzureOpenAI(
    api_key = os.getenv("DEPLOYMENT_OPENAI_API_KEY"), 
    #api_version="2024-02-01",
    api_version = "2024-03-01-preview",
    azure_endpoint = os.getenv("DEPLOYMENT_OPENAI_API_BASE") # Use /v1/ if needed
)
data = json.load(open('avg_bertscore.json'))

#subject = 'man'
#subject = 'humanoid robot'
#subject = 'woman'
for subject in ['human aged under 18 years', 'human aged 18-44 years', 'human aged 45-64 years', 'human aged over 65 years', 'African American human', 'Caucasian human', 'Asian human', 'man', 'woman', 'humanoid robot']:
    ROOT_DIR = f'/home/t-ashsathe/BlobStorage/containers/absathe/MultiModalBias/NOFILTER/{subject.replace(" ", "_")}'

    os.makedirs(ROOT_DIR, exist_ok=True)

    for item in tqdm(data):
        profession = item['gold_occupation'].replace('/', ' or ')
        #prompt = item['predictions'][0]['prompt'].replace('<subject>', subject)
        prompt = list(item['predictions'].keys())[0].replace('<subject>', subject)
        save_path = os.path.join(ROOT_DIR, profession + '.png')
        if not os.path.exists(save_path):
            try:
                prompt = TEMPLATE.format(profession=profession, prompt=prompt)
                image_url = request_dalle3(client, prompt, model='Dalle3')
            except Exception as e: 
                logger.exception(f'Error on prompt "{prompt}"')
                continue
            else:
                urllib.request.urlretrieve(image_url, save_path)
