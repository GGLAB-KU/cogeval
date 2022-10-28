
import numpy as np
import pandas as pd
import yaml, argparse, os
from yaml.loader import SafeLoader
from scipy.special import softmax
from scipy.stats import entropy
from common import *

def mc_dropout(data, model, tokenizer, iters):
  sm_sum = dict()
  for counter in range(iters):
      print('Iteration: ', counter)
      correct = 0
      for ind, row in data.iterrows():
          text = row['content']
          encoded_input = tokenizer(text, return_tensors='pt')
          encoded_input = encoded_input.to('cuda')
          output = model(**encoded_input)
          scores = output[0][0].detach().cpu() #.numpy()
          scores = softmax(scores)
          pred = np.argmax(scores)
          guid = row['sstb_id'] 
          if guid in sm_sum:
              for i in range(len(scores)):
                sm_sum[guid][i] += scores[i]
          else:
            sm_sum[guid] = scores.copy()
          if row['sstb_label'] == pred:
            correct += 1
  return sm_sum


def get_metrics(df, probs, iters):
  pred_confidences = dict()
  gold_confidences = dict()
  uncertainty =  dict()

  out_df =  pd.DataFrame({})
  for guid in probs.keys():
    
    # avg over iterations 
    probs[guid] = probs[guid] / iters

    # get confidence score of model's prediction
    pred_confidences[guid] = max(probs[guid])
    
    # get uncertainty over model's prediction
    uncertainty[guid] = entropy(probs[guid], base=2)

    # append entropy values to the data
    guid_row = df.loc[df['sstb_id'] == guid]

    # get confidence score for ground truth
    gold_label = guid_row.iloc[0]['sstb_label']
    gold_confidences[guid] = probs[guid][gold_label]

    df2 = guid_row[['sstb_id','zuco_id','content','control_id','sstb_label']]
    df2['pred_confidence'] = pred_confidences[guid]
    df2['gold_confidence'] = gold_confidences[guid]
    df2['uncertainty'] = uncertainty[guid]

    out_df = pd.concat([out_df, df2])  
  return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # load data
    data_path      = config['data']['file'] 
    data = load_data(data_path)

    # inference with model after calibration with mc_dropout
    model_name     = config['model']['name']
    num_iters      = config['model']['calibration']['num_iters']
    model, tokenizer = load_model(model_name)
    model = model.to('cuda')

    sft_values = mc_dropout(data, model, tokenizer, num_iters)
    results   = get_metrics(data, sft_values, num_iters)
  
  
    # save metric results
    results_dir  = config['results']['dir']
    results_file  = config['results']['file']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results.to_csv(results_dir+'/'+results_file)

if __name__=="__main__":
    # specify parameters in .yaml file under ./config/metrics directory 
    # usage: python mc_dropout.py <config_file>
    # e.g. : python mc_dropout.py config/metrics/machine/sa-machine_1.yaml 
    main()
    