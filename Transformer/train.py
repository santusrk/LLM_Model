import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

def get_all_sentences(ds,lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # here convert sentence into work or token using Wordlevel
        tokenizer.pre_tokenizer=Whitespace() # here seperate the word into whitespace
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"] , min_frequency = 2) # cetain rulelike collect the word that have atlest 2 times repeated word is there rest ignore
        tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer) # here tokenizer learn to take all the sentence line by line and collect the most frequecy word to frame the new sentance
        tokenizer.save(str(tokenizer_path)) # save the tokenizer into the given path
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train')

    # build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # split 90% train and 10% test
    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw,val_ds_raw = random_split(ds_raw,[train_ds_size,val_ds_size])







