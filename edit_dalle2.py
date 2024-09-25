from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
from PIL import Image
from glob import glob
from tqdm import tqdm
import io
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
in_dir = '/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/dall-e-3/'
out_dir = '/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/edits/dall-e-2/'
with open('./final_fnames.txt', 'r') as f:
    fnames = [l.strip() for l in f.readlines()]

editing_prompt = 'Edit the blue humanoid robot in this image into a human person. The description is "{desc}". Ensure NO robots in the output'

@backoff.on_exception(backoff.expo, Exception, max_time=600)
def request_dalle2(
    client, prompt, image_path, model='dall-e-2', size='1024x1024', n=1
): 
    image = Image.open(image_path).convert('RGBA')
    image.putalpha(0)
    bytesio = io.BytesIO()
    image.save(bytesio, 'png')
    image_bytes = bytesio.getvalue()
    response = client.images.edit(
        model=model,
        prompt=prompt,
        image=image_bytes,
        size=size,
        n=n,
    )
    return response.data[0].url

client = OpenAI()
os.makedirs(out_dir, exist_ok=True)
for fname in tqdm(fnames):
    out_fname = fname.replace('blue humanoid robot', 'person')
    in_fname = os.path.join(in_dir, fname)
    save_path = os.path.join(out_dir, out_fname)
    if not os.path.exists(save_path):
        try:
            image_url = request_dalle2(client, fname.rstrip('.png').replace('blue humanoid robot', 'person'), in_fname)
        except Exception as e: 
            logger.exception(f'Error on prompt "{fname}"')
            continue
        else:
            urllib.request.urlretrieve(image_url, save_path)