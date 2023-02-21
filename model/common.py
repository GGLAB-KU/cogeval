from plistlib import load
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import pandas as pd

def load_model(name):
  tokenizer = AutoTokenizer.from_pretrained(name)
  model = AutoModelForSequenceClassification.from_pretrained(name)
  #model.to('cuda')
  model.train()
  return model, tokenizer


def load_data(data):
  df = pd.read_csv(data, sep=',', header=0)
  df['content'] = df['content'].str.lower()
  df = df.astype({'sstb_label':'int'})
  return df

def load_data_lalor(data):
  df = pd.read_csv(data, sep=',', header=0)
  df['content'] = df['content'].str.lower()
  df = df.astype({'sst_phrase_id':'int'})
  return df