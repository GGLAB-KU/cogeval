from itertools import compress
from sklearn.metrics import jaccard_score
import pandas as pd

SET_ID = "userstudy"
# read model probs and preds
models = ['davinci', 'lstm', 'random','roberta', 'tfidf', 'human']
model_probs = dict()
model_predictions = dict()
for model in models:
    df = pd.read_csv("results/language-inference/lalor/files/"+model+".csv")
    model_probs[model] = df[['e', 'n', 'c']].values.tolist()   
    model_predictions[model] = df['pred_label'].tolist()

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
jaccard_df.to_csv("results/language-inference/lalor/jaccard_sim_"+SET_ID+".csv", header=True, index=False, sep=',')