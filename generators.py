import os
import torch
import backoff
import urllib.request
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI, AzureOpenAI
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM, pipeline
load_dotenv('multimodalbias.env')

def setup_model(model_name):
    name_to_client = {
        'llava': get_llava_client,
        'vipllava': get_vipllava_client,
        'gemini': get_gemini_client,
        'gemini_vision': get_gemini_vision_client,

        'llama': get_llama_client, 
    }
    name_to_request = {
        'llava': request_llava,
        'vipllava': request_vipllava,
        'gemini': request_gemini,
        'gemini_vision': request_gemini_vision,

        'llama': request_llama,
    }
    return name_to_client[model_name](), name_to_request[model_name]

def get_openai_client():
    client = OpenAI()
    return client

def get_azureopenai_client():
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"), 
        #api_version="2024-02-01",
        api_version = "2024-03-01-preview",
        azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE") # Use /v1/ if needed
    )
    #openai.api_type = "azure"
    #openai.api_version = "2023-05-15"

    return client

def get_gemini_client():
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    client = genai.GenerativeModel('gemini-pro')
    return client

def get_gemini_vision_client():
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    client = genai.GenerativeModel('gemini-pro-vision')
    return client

def get_llava_client():
    model_id = 'llava-hf/llava-v1.6-mistral-7b-hf'
    return {
        'processor': AutoProcessor.from_pretrained(model_id),
        'model': AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    }

def get_vipllava_client():
    model_id = 'llava-hf/vip-llava-7b-hf'
    return {
        'processor': AutoProcessor.from_pretrained(model_id),
        'model': AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    }

def get_llama_client():
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    return pipeline(
        'text-generation',
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map='cuda'
    ) 

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

@backoff.on_exception(backoff.expo, Exception, max_time=600)
def request_gpt4(
    client, prompt, model='gpt-4-32k', max_tokens=1024
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

@backoff.on_exception(backoff.expo, Exception, max_time=600)
def request_gemini(model, text_prompt):
    response = model.generate_content([text_prompt], stream=True)
    response.resolve()
    return response.text

@backoff.on_exception(backoff.expo, Exception, max_time=600)
def request_gemini_vision(model, text_prompt, img):
    response = model.generate_content([text_prompt, img], stream=True)
    response.resolve()
    return response.text

def request_llava(model, text_prompt, img):
    reformatted_prompt = f'[INST] <image>\n{text_prompt} [/INST]'
    inputs = model['processor'](reformatted_prompt, img, return_tensors='pt').to('cuda')
    output = model['model'].generate(**inputs, max_new_tokens=300)
    num_toks = inputs.input_ids.size(1)
    return model['processor'].decode(output[0][num_toks:], skip_special_tokens=True)

def request_vipllava(model, text_prompt, img):
    reformatted_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\n{text_prompt}###Assistant:"
    inputs = model['processor'](reformatted_prompt, img, return_tensors='pt').to('cuda')
    output = model['model'].generate(**inputs, max_new_tokens=300)
    num_toks = inputs.input_ids.size(1)
    return model['processor'].decode(output[0][num_toks:], skip_special_tokens=True)

def request_llama(model, text_prompt, img=None):
    return model(text_prompt, num_return_sequences=1, max_new_tokens=10)[0]['generated_text'].replace(text_prompt, '')