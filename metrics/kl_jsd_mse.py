import pandas as pd
import numpy as np
from itertools import compress
from tensorflow import keras
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error


SET_ID = "v3_fdf95_added"
#df_ids = pd.read_csv("results/SNLI-lalor/v2_USERSTUDY.csv")
df = pd.read_csv("results/SNLI-lalor/"+SET_ID+".csv")
df_ids = df
subset_ids = (df_ids.sample_id).tolist()

gold_labels = df['gold_label'].tolist()
num_samples = len(subset_ids)

print("SET: %s" % SET_ID)
print("num samples: %d" % num_samples )


def calc_dist(p, q):
    kl = keras.losses.KLDivergence()    
    kl = np.mean(kl(p, q).numpy())    
    jsd = np.mean([distance.jensenshannon(p[i], q[i]) for i in range(len(p))])    
    mse = mean_squared_error(p, q)
    return {"KL": kl, "JSD": jsd, "MSE": mse}


models = ['random_noisy', 'tfidf', 'lstm_calibrated','roberta_calibrated', 'davinci', 'human', 'lstm', 'roberta']
model_probs = dict(); model_predictions = dict()
dist_labels = ['KL', "JSD", 'MSE']
subsets = ['ALL', 'AGREE']#, 'DISAGREE']
results = {"ALL": dict(), "AGREE": dict()}#, "DISAGREE": dict()}
model_1 = []; model_2 = []
agree_count = []; disagree_count = []

# read model probs and preds
for model in models:
    model_probs[model] = []
    model_predictions[model] = []
    mdf = pd.read_csv("data/machine/SNLI-lalor/"+"v3_all"+"/"+model+".csv")
    for idx, row in mdf.iterrows():
        if int(row['sample_id'])  in subset_ids:
            model_probs[model].append(row[['e', 'n', 'c']].values.tolist())
            model_predictions[model].append(row['pred_label'])
    print(len(model_predictions[model]))


# calculate pairwise distances
for i in range(len(models)):
    for j in range(len(models)):
        if i != j:
            model_1.append(models[i])
            model_2.append(models[j])
            dist = calc_dist(model_probs[models[i]], model_probs[models[j]])
            results['ALL'][f"{models[i]}/{models[j]}"] = dist
            
            # agreements
            agreement_mask = [pi==pj for pi, pj in zip(model_predictions[models[i]], model_predictions[models[j]])]            
            probs_model_1 = list(compress(model_probs[models[i]], agreement_mask))
            probs_model_2 = list(compress(model_probs[models[j]], agreement_mask))
            agree_count.append(round(len(probs_model_1)/num_samples, 2))
            dist = calc_dist(probs_model_1, probs_model_2)           
            results['AGREE'][f"{models[i]}/{models[j]}"] = dist            
            
            '''# disagreements
            disagreement_mask = np.logical_not(agreement_mask)
            probs_model_1 = list(compress(model_probs[models[i]], disagreement_mask))
            probs_model_2 = list(compress(model_probs[models[j]], disagreement_mask))
            disagree_count.append(round(len(probs_model_1)/num_samples, 2))
            dist = calc_dist(probs_model_1, probs_model_2)           
            results['DISAGREE'][f"{models[i]}/{models[j]}"] = dist'''

## dump to file
result_df = pd.DataFrame()
result_df['model_1'] = model_1
result_df['model_2'] = model_2
for subset in subsets:
    for metric in dist_labels:
        col_data = []
        for pair in results[subset].keys():
            col_data.append(results[subset][pair][metric])
        if subset == 'AGREE':
            result_df['AGREE_PCT'] = agree_count
        elif subset == 'DISAGREE':
            result_df['DISAGREE_PCT'] = disagree_count
        result_df[f"{subset}_{metric}"] = col_data       
result_df.to_csv("results/SNLI-lalor/"+SET_ID+"/distances/kl-jsd-mse_dist.csv", header=True, index=False, sep=',')


## Per label distance
'''labels = ["contradiction", "neutral", "entailment"]
results_per_label = {label: dict() for label in labels}
model_1 = []; model_2 = []
for i in range(len(models)):
    for j in range(len(models)):
        if i != j:
            model_1.append(models[i])
            model_2.append(models[j])
            for label in labels:           
              label_mask = [gold_labels[i] == label for i in range(len(gold_labels))]            
              probs_model_1 = list(compress(model_probs[models[i]], label_mask))
              probs_model_2 = list(compress(model_probs[models[j]], label_mask))
              dist = calc_dist(model_probs[models[i]], model_probs[models[j]])
              results_per_label[label][f"{models[i]}/{models[j]}"] = dist         


result_per_label_df = pd.DataFrame()
result_per_label_df['model_1'] = model_1
result_per_label_df['model_2'] = model_2
for label in labels:
    for metric in dist_labels:
        col_data = []
        for pair in results_per_label[label].keys():
            col_data.append(results_per_label[label][pair][metric])
        result_per_label_df[f"{label}_{metric}"] = col_data
result_per_label_df.to_csv("results/SNLI-lalor/v2/distances/"+SET_ID+"_kl-jsd-mse_per_label_dist.csv", header=True, index=False, sep=',')'''