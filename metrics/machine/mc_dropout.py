
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
          guid = row['sst_phrase_id'] 
          if guid in sm_sum:
              for i in range(len(scores)):
                sm_sum[guid][i] += scores[i]
          else:
            sm_sum[guid] = scores.copy()
          if row['three_way_labels'] == pred-1:
            correct += 1
  print("#correct: ", correct)
  print("#total: ", len(sm_sum))
  return sm_sum

def get_metrics_lalor(df, probs, iters):
  pred_confidences = dict()
  gold_confidences = dict()
  uncertainty =  dict()

  out_df =  pd.DataFrame({})
  out_false_df =  pd.DataFrame({})
  out_true_df =  pd.DataFrame({})
  for guid in probs.keys():
    
    # avg over iterations 
    probs[guid] = probs[guid] / iters
    # get confidence score of model's prediction
    pred_confidences[guid] = max(probs[guid])
    # get argmax
    pred_label = np.argmax(probs[guid]) - 1

    # get uncertainty over model's prediction
    uncertainty[guid] = entropy(probs[guid], base=2)

    # append entropy values to the data
    guid_row = df.loc[df['sst_phrase_id'] == guid]

    # get confidence score for ground truth
    gold_label = guid_row.iloc[0]['three_way_labels']
    
    gold_confidences[guid] = probs[guid][gold_label+1]

    df2 = guid_row[['sample_id', 'sst_phrase_id','sst_sentence_id','content','three_way_labels', "average_accuracy", "flesch_score_textstat", "mean_grade_level_textstat", "number_of_words","number_of_characters"]]
    df2['pred_confidence'] = pred_confidences[guid]
    df2['pred_label'] = pred_label
    df2['gold_label'] = gold_label
    df2['gold_confidence'] = gold_confidences[guid]
    df2['uncertainty'] = uncertainty[guid]

    if pred_label != gold_label:
      false_cases = guid_row[['sample_id', 'sst_phrase_id','sst_sentence_id','content','three_way_labels', "average_accuracy", "flesch_score_textstat", "mean_grade_level_textstat", "number_of_words","number_of_characters"]]
      false_cases['pred_label'] = pred_label
      false_cases['gold_label'] = gold_label
      false_cases['pred_confidence'] = pred_confidences[guid]
      false_cases['gold_confidence'] = gold_confidences[guid]
      false_cases['uncertainty'] = uncertainty[guid]
      out_false_df = pd.concat([out_false_df, false_cases])
    else:
      true_cases = guid_row[['sample_id', 'sst_phrase_id','sst_sentence_id','content','three_way_labels', "average_accuracy", "flesch_score_textstat", "mean_grade_level_textstat", "number_of_words","number_of_characters"]]
      true_cases['pred_label'] = pred_label
      true_cases['gold_label'] = gold_label
      true_cases['pred_confidence'] = pred_confidences[guid]
      true_cases['gold_confidence'] = gold_confidences[guid]
      true_cases['uncertainty'] = uncertainty[guid]
      out_true_df = pd.concat([out_true_df, true_cases])
        
    out_df = pd.concat([out_df, df2])  
  return out_df, out_false_df, out_true_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # load data
    data_path      = config['data']['file'] 
    data = load_data_lalor(data_path)

    # inference with model after calibration with mc_dropout
    model_name     = config['model']['name']
    num_iters      = config['model']['calibration']['num_iters']
    model, tokenizer = load_model(model_name)
    model = model.to('cuda')

    sft_values = mc_dropout(data, model, tokenizer, num_iters)
    results, false_results, true_results  = get_metrics_lalor(data, sft_values, num_iters)
  
  
    # save metric results
    results_dir  = config['results']['dir']
    results_file  = config['results']['file']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results.to_csv(results_dir+'/'+results_file, index = False)
    false_results.to_csv(results_dir+'/'+'falses.csv', index = False)
    true_results.to_csv(results_dir+'/'+'trues.csv', index = False)

if __name__=="__main__":
    # specify parameters in .yaml file under ./config/metrics directory 
    # usage: python mc_dropout.py <config_file>
    # e.g. : python mc_dropout.py config/metrics/machine/sa-machine_1.yaml 
    main()
    