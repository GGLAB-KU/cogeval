import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from scipy.stats import entropy
from common import *

def T_scaling(logits, temperature):
  return torch.div(logits, temperature)


def learn_T(data, model, tokenizer):
  logits_list = []
  labels_list = []
  for ind, row in data.iterrows():
      text = row['sentence']
      encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
      output = model(**encoded_input)
      # logits = output[0][0].detach().numpy()
      logits = output[0][0]
      label = torch.Tensor([row['label']])
      model.eval()
      with torch.no_grad():
        logits_list.append(logits)
        labels_list.append(label)
      del logits
      del output
  logits_list = torch.stack(logits_list).to('cuda')
  labels_list = torch.cat(labels_list).to('cuda')
  logits_list = logits_list.detach()
  labels_list = labels_list.type(torch.LongTensor).to('cuda')
  temps = []
  losses = []
  def _eval():
    loss = criterion(T_scaling(logits_list, temperature), labels_list)
    loss.backward()
    temps.append(temperature.item())
    losses.append(loss)
    return loss
  temperature = nn.Parameter(torch.ones(1).cuda())
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
  optimizer.step(_eval)
  return temperature.item()



def scale_with_T(temperature, data, model, tokenizer):
  sample_entropy_t_scaled = {}
  sample_entropy = {}
  model.eval()
  for ind, row in data.iterrows():
      text = row['content']
      encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
      output = model(**encoded_input)
      logits = output[0][0].detach().cpu().numpy()
      logits_scaled = logits / temperature.item()
      # logits_scaled = logits / 2.0
      print(logits, '\t', logits_scaled)
      scores = softmax(logits)
      scores_scaled = softmax(logits_scaled)
      sample_entropy[row['sstb_id']] = entropy(scores, base=2)
      sample_entropy_t_scaled[row['sstb_id']] = entropy(scores_scaled, base=2)
      print(scores, '\t', scores_scaled)
      print(sample_entropy[row['sstb_id']], '\t', sample_entropy_t_scaled[row['sstb_id']])
      print('=====================================================================')
      del scores
      del output
  return sample_entropy, sample_entropy_t_scaled


if __name__=="__main__":
  model, tokenizer = load_model('Souvikcmsa/BERT_sentiment_analysis')
  dev_data = pd.read_csv('data/human/sentiment-analysis/dev.tsv', sep='\t', header=0)
  temperature = learn_T(dev_data, model, tokenizer)
  data = load_data('data/human/sentiment-analysis/all.csv')
  sample_entropy, sample_entropy_t_scaled = scale_with_T(temperature, data, model, tokenizer)
  for guid in sample_entropy.keys():
    print(guid, '\t', sample_entropy[guid], '\t', sample_entropy_t_scaled[guid])