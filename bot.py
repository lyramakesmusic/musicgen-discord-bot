import discord
from dotenv import load_dotenv

import torch, torchaudio
from audiocraft.data.audio import audio_write
import audiocraft.models
import time
import os

import gdown
import shutil

import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

bot = discord.Bot()
executor = ThreadPoolExecutor()

# generate from small model
@bot.command(description="Generates a 10s sample with base musicgen-small")
async def generate(ctx, prompt: str):

    loop = asyncio.get_event_loop()
    await ctx.defer()
    
    seed = int(time.time())
    torch.manual_seed(seed)
    
    musicgen = await loop.run_in_executor(executor, audiocraft.models.MusicGen.get_pretrained, 'facebook/musicgen-small')
    musicgen.set_generation_params(duration=10)
    
    t0 = time.time()
    wav = await loop.run_in_executor(executor, musicgen.generate, [prompt])

    fname = "_".join([prompt.replace(" ", "_")[:50], str(seed)])
    audio_write(f'outputs/{fname}', wav[0].cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f'saved {fname}.wav')

    del musicgen
    torch.cuda.empty_cache()

    await ctx.followup.send(f"generated `{prompt}` in {round(time.time()-t0, 2)}s", file=discord.File(f'outputs/{fname}.wav'))

# generate from medium model
@bot.command(description="Generates a 10s sample with base musicgen-medium")
async def generate_medium(ctx, prompt: str):

    loop = asyncio.get_event_loop()
    await ctx.defer()
    
    seed = int(time.time())
    torch.manual_seed(seed)
    
    musicgen = await loop.run_in_executor(executor, audiocraft.models.MusicGen.get_pretrained, 'facebook/musicgen-medium')
    musicgen.set_generation_params(duration=10)
    
    t0 = time.time()
    wav = await loop.run_in_executor(executor, musicgen.generate, [prompt])

    fname = "_".join(["medium", prompt.replace(" ", "_")[:50], str(seed)])
    audio_write(f'outputs/{fname}', wav[0].cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f'saved {fname}.wav')

    del musicgen
    torch.cuda.empty_cache()

    await ctx.followup.send(f"generated `{prompt}` in {round(time.time()-t0, 2)}s", file=discord.File(f'outputs/{fname}.wav'))

# generate from specified finetuned model
@bot.command(description="Generates a 10s sample with the given finetune")
async def generate_finetuned(ctx, prompt: str, model: str):

    loop = asyncio.get_event_loop()
    await ctx.defer()
    
    seed = int(time.time())
    torch.manual_seed(seed)

    try:
        musicgen = await loop.run_in_executor(executor, audiocraft.models.MusicGen.get_pretrained, f'checkpoints/{model}')
        musicgen.set_generation_params(duration=10)
    except:
        await ctx.followup.send(f"couldn't load `{model}`, try downloading it first- `/get_checkpoint model_name:{model} state_dict_gdrive_url:shared/google/drive/url/to/state_dict.bin`")
        return

    t0 = time.time()
    wav = await loop.run_in_executor(executor, musicgen.generate, [prompt])

    fname = "_".join([model.replace(" ", "_"), prompt.replace(" ", "_")[:50], str(seed)])
    audio_write(f'outputs/{fname}', wav[0].cpu(), musicgen.sample_rate, strategy="loudness", loudness_compressor=True)
    print(f'saved {fname}.wav')

    del musicgen
    torch.cuda.empty_cache()

    await ctx.followup.send(f"generated `{prompt}` in {round(time.time()-t0, 2)}s", file=discord.File(f'outputs/{fname}.wav'))

# download finetuned model from google drive link
@bot.command(description="Saves a finetuned state_dict.bin to use for inference")
async def get_checkpoint(ctx, model: str, state_dict_gdrive_url: str):

    loop = asyncio.get_event_loop()

    await ctx.defer()
    if os.path.isdir(f"checkpoints/{model}"):
        await ctx.followup.send(f"model `{model}` already exists, skipping download. please use a new name!")
        return

    delay_msg = await ctx.send("this will take several minutes to download. please be patient!")

    dir_path = f'checkpoints/{model}'
    file_path = f'{dir_path}/state_dict.bin'
    os.makedirs(dir_path, exist_ok=True)

    file_id = state_dict_gdrive_url.split('/')[-2]
    t0 = time.time()
    try:
        await loop.run_in_executor(executor, gdown.download, f'https://drive.google.com/uc?id={file_id}', file_path, False)
    except:
        await ctx.followup.send(f"Can't download model from url - requires a public google drive link to `state_dict.bin`")
        return

    await loop.run_in_executor(executor, shutil.copy2, 'checkpoints/compression_state_dict.bin', f'checkpoints/{model}/compression_state_dict.bin')

    print(f'saved checkpoints/{model}')
    await ctx.followup.send(f"saved checkpoint in {round(time.time()-t0, 2)}s\nrun `/generate_finetuned prompt:prompt model:{model}`")
    await delay_msg.delete()

print('logged in')
bot.run(TOKEN)
