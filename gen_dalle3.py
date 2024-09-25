from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
from tqdm import tqdm
import os
import backoff
import urllib.request
load_dotenv('envs/openai.env')
"""
def run_gpt4v(
    instruction,
    image=None,
    client=None,
    max_tokens=300,
    temperature=0.0,
    model_name="gpt-4-vision-preview",
):
    messages = [{"role": "user", "content": [instruction, {"image": image}]}]
    params = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    c = 1
    while c < 3000:
        try:
            response = client.chat.completions.create(**params)
            sleep(0.5)
            break
        except openai.RateLimitError as e:
            c *= 2
            sleep(c)
    return response.choices[0].message.content
buffer = BytesIO()
image.save(buffer, format="JPEG")
encoded_image = b64encode(buffer.getvalue()).decode("utf-8")

output[lang_code] = run_gpt4v(
    instruction=instruction,
    image=encoded_image,
    client=client,
    model_name=args.model,
    temperature=args.temperature,
    max_tokens=args.max_new_tokens,
)
"""
out_dir = '/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/text-to-image/dall-e-3/'
with open('./final_fnames.txt', 'r') as f:
    prompts = [l.strip() for l in f.readlines()]

@backoff.on_exception(backoff.expo, Exception, max_time=600)
def request_dalle3(
    client, prompt, model='dall-e-3', size='1024x1024', quality='standard', n=1
):    
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )
    return response.data[0].url

client = OpenAI()
for prompt in tqdm(prompts):
    prompt = prompt.rstrip('.png')
    prompt = prompt.replace('.', '').replace('blue humanoid robot', 'human person')
    save_path = os.path.join(out_dir, prompt + '.png')
    if not os.path.exists(save_path):
        try:
            image_url = request_dalle3(client, prompt)
        except Exception as e: 
            logger.exception(f'Error on prompt "{prompt}"')
            continue
        else:
            urllib.request.urlretrieve(image_url, save_path)