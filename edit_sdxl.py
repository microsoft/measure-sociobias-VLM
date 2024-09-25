from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
from tqdm import tqdm
from PIL import Image
import torch
out_dir = '../BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/edits/SDXL'
in_dir = '../BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/dall-e-3'
device = 'cuda'
editing_prompt = 'Edit the blue humanoid robot in this image into a human person. The description is "{desc}". Ensure NO robots in the output'
editing_prompt = 'The description is "{desc}"'
#negative_prompt = 'generic, out of picture, cropped, low detail environment, low detail background'
negative_prompt = 'robots, empty scene, bad background, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

base = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', 
    torch_dtype=torch.float16, 
    variant='fp16', 
    use_safetensors=True,
).to(device)

pipeline = AutoPipelineForImage2Image.from_pipe(base).to(device)
with open('./final_fnames.txt', 'r') as f:
    fnames = [line.strip() for line in f.readlines()]

for fname in tqdm(fnames):
    actual_prompt = editing_prompt.format(
        desc=fname.rstrip('.png').replace('blue humanoid robot', 'human person')
    )
    out_fname = f'{out_dir}/{fname.replace("blue humanoid robot", "person")}'
    init_image = Image.open(f'{in_dir}/{fname}') 
    print(actual_prompt)
    image = pipeline(
        actual_prompt,
        image=init_image,
        strength=0.8,
        guidance_scale=10.5
    ).images[0]
    image.save(out_fname)