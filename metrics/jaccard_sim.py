from itertools import compress
from sklearn.metrics import jaccard_score
import pandas as pd

SET_ID = "v3_all"
#df_ids = pd.read_csv("user-study/questions/v2_tilek_greedy.csv")
df = pd.read_csv("results/SNLI-lalor/"+SET_ID+".csv")
df_ids = df
subset_ids = (df_ids.sample_id).tolist()


# read model probs and preds
models = ['random_noisy', 'tfidf', 'lstm','roberta_calibrated', 'davinci', 'human']

model_probs = dict()
model_predictions = dict()
for model in models:
    model_probs[model] = []
    model_predictions[model] = []
    mdf = pd.read_csv("data/machine/SNLI-lalor/"+"v3_all"+"/"+model+".csv")
    for idx, row in mdf.iterrows():
        if int(row['sample_id'])  in subset_ids:
            model_probs[model].append(row[['e', 'n', 'c']].values.tolist())
            model_predictions[model].append(row['pred_label'])

    print(len(model_predictions[model]))
scoring_modes = ['micro', 'macro', 'weighted']
results_jaccard = {mode: dict() for mode in scoring_modes}
model_1 = []; model_2 = []
for i in range(len(models)):
    for j in range(len(models)):
        if i != j:
            model_1.append(models[i])
            model_2.append(models[j])
            dist = jaccard_score(model_predictions[models[i]], model_predictions[models[j]], average='micro')
            results_jaccard['micro'][f"{models[i]}/{models[j]}"] = dist          
            dist = jaccard_score(model_predictions[models[i]], model_predictions[models[j]], average='macro')
            results_jaccard['macro'][f"{models[i]}/{models[j]}"] = dist          
            dist = jaccard_score(model_predictions[models[i]], model_predictions[models[j]], average='weighted')
            results_jaccard['weighted'][f"{models[i]}/{models[j]}"] = dist   


jaccard_df = pd.DataFrame()
jaccard_df['model_1'] = model_1
jaccard_df['model_2'] = model_2
for mode in scoring_modes:
    jaccard_df[mode] = list(results_jaccard[mode].values())
jaccard_df.to_csv("results/SNLI-lalor/"+SET_ID+"/distances/"+"jaccard_sim.csv", header=True, index=False, sep=',')
