
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy
from common import *

def mc_dropout(data, model, tokenizer, iters):
  num_classes = 3
  sm_sum = dict()
  for counter in range(iters):
      print('Iteration: ', counter)
      correct = 0
      for ind, row in data.iterrows():
          text = row['content']
          encoded_input = tokenizer(text, return_tensors='pt')
          output = model(**encoded_input)
          scores = output[0][0].detach().numpy()
          scores = softmax(scores)
          pred = np.argmax(scores)
          guid = row['sstb_id'] 
          if guid in sm_sum:
              for i in range(num_classes):
                sm_sum[guid][i] += scores[i]
          else:
            sm_sum[guid] = scores.copy()
          if row['sstb_label'] == pred:
            correct += 1
  return sm_sum

def entropy_over_softmax(df, probs, iters):
  sample_entropy = dict()
  out_df =  pd.DataFrame({})
  for guid in probs.keys():
    # avg over iterations 
    probs[guid] = probs[guid] / iters
    sample_entropy[guid] = entropy(probs[guid], base=2)
    # append entropy values to the data
    guid_row = df.loc[df['sstb_id'] == guid]
    df2 = guid_row[['sstb_id','zuco_id','content','control_id','sstb_label']]
    df2['BERT_MC_DROPOUT'] = sample_entropy[guid]
    out_df = pd.concat([out_df, df2])  
  return out_df


if __name__=="__main__":
  iters = 50
  data = load_data('data/human/sentiment-analysis/all.csv')
  model, tokenizer = load_model('Souvikcmsa/BERT_sentiment_analysis')
  sft_values = mc_dropout(data, model, tokenizer, iters)
  results   = entropy_over_softmax(data, sft_values, iters)
  results.to_csv('results/sentiment-analysis/mc_dropout.csv', index=False)