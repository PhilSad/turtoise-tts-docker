# Imports used through the rest of the notebook.
from os import fchmod
import click
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import pysbd

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()
seg = pysbd.Segmenter(language="en", clean=False)




def split_long_text(text, max_words_per_sentence = 50):

  sentences = seg.segment(text)
  sentences_split = [e.split(' ') for e in sentences]
  
  batches = [[]]  
  cur_i_batches = 0
  cur_len = 0
  for sentence_split in sentences_split:
    if cur_len + len(sentence_split) > max_words_per_sentence:
      cur_i_batches += 1
      batches.append([' '.join(sentence_split).strip()])
      cur_len = 0
    else:
      batches[cur_i_batches].append(' '.join(sentence_split).strip())
      cur_len += len(sentence_split)
  
  return batches
 

def read(text, voice="train_atkins", preset="ultra_fast", prefix=""):
  save_folder = './vol/output/'
  text_list = text.split('\n')
  text_list = [t for t in text_list if t != '' and t!=' ']

  for i,t in enumerate(text_list):
    
    todos = split_long_text(t)

    for j,batch in enumerate(todos):
      text = ' '.join(batch)
      print(text)
      # Load it and send it through Tortoise.
      voice_samples, conditioning_latents = load_voice(voice)
      gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                                preset=preset)
      save_path=f'{save_folder}/{prefix}_{str(i)}_{str(j)}.wav'
      torchaudio.save(save_path, gen.squeeze(0).cpu(), 24000)
      print(f'saving ... {text}')


@click.command()
@click.option('--path', default='./vol/input/input.txt', help='input file')
@click.option('--voice', default='angie', help='wich voice should be used to read the file')
@click.option('--preset', default='ultra_fast', help='speed of inference: ultra_fast, fast, standard')
def read_command(path, voice, preset):
  # TODO: read all .txt files
  with open('./vol/input/input.txt') as f:
    text = f.read()
  read(text, voice, preset)
  
if __name__ == '__main__':
  read_command()







