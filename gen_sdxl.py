from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
from tqdm import tqdm
from PIL import Image
import torch
out_dir = '../BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/text-to-image/SDXL'
device = 'cuda'
additional_prompt = '{prompt}, muted colors, wide shot'
#negative_prompt = 'generic, out of picture, cropped, low detail environment, low detail background'
negative_prompt = 'robots, empty scene, bad background, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature'

base = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', 
    torch_dtype=torch.float16, 
    variant='fp16', 
    use_safetensors=True,
).to(device)

refiner = DiffusionPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant='fp16',
).to(device)

with open('./final_fnames.txt', 'r') as f:
    fnames = [line.strip() for line in f.readlines()]

for fname in tqdm(fnames):
    actual_prompt = additional_prompt.format(
        prompt=fname.rstrip('.png').replace('blue humanoid robot', 'human person')
    )
    out_fname = f'{out_dir}/{fname.replace("blue humanoid robot", "human person")}'
    print(actual_prompt)
    image = base(
        prompt=actual_prompt, 
        output_type='latent', 
        negative_prompt=negative_prompt, 
        guidance_scale=6,
        num_inference_steps=48,
    ).images[0]
    image = refiner(
        prompt=actual_prompt, 
        image=image[None, :], 
        negative_prompt=negative_prompt, 
        guidance_scale=6,
        num_inference_steps=28,
    ).images[0]
    image.save(out_fname)