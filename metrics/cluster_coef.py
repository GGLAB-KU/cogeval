import pandas as pd
import numpy as np
from itertools import compress
from tensorflow import keras
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error
import networkx as nx
from collections import defaultdict

SET_ID = "all-set"
models = ['random', 'tfidf', 'lstm', 'roberta', 'davinci', 'human']
#models.sort()
model_probs = dict()
model_predictions = dict()


sf = pd.read_csv('results/language-inference/lalor/SNLI_results_'+SET_ID+'.csv')

def calc_dist(pr, qr):
    if len(pr)!=1 or len(qr)!=1:
        breakpoint()
    kl = keras.losses.KLDivergence()    
    kl = np.mean(kl(pr, qr).numpy())    
    jsd = np.mean([distance.jensenshannon(pr[i], qr[i]) for i in range(len(pr))])    
    mse = mean_squared_error(pr, qr)
    return 1-jsd
 


adf = pd.DataFrame({'sample_id':[], 'avg_coef':[], 'all_coefs':[]})

all_of_all =  defaultdict(lambda:0)
all_of_avgs = 0
for idx, row in sf.iterrows():
    sid = row['sample_id']
    G = nx.Graph()
    G.add_nodes_from([(0, {"model": "random"}),(1, {"model": "tfidf"}),(2, {"model": "lstm"}),(3, {"model": "roberta"}),(4, {"model": "davinci"}),(5, {"model": "human"})])

    #print(sid)
    for model in models:
        df = pd.read_csv('results/language-inference/lalor/files/'+model+'.csv')
        df = df[df.sample_id.isin([sid])]
        model_probs[model] = df[['e', 'n', 'c']].values.tolist()   
        model_predictions[model] = df['pred_label'].tolist()


    model_1 = []; model_2 = []
    added = []
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                #print(models[i], models[j])
                model_1.append(models[i])
                model_2.append(models[j])
                if (j,i) not in added:
                    dist = calc_dist(model_probs[models[i]], model_probs[models[j]])
                    G.add_weighted_edges_from([(i, j, dist)])
                added.append((i,j))
    avg_coef = nx.average_clustering(G, weight='weight')
    all_coefs = nx.clustering(G, weight='weight')
    for k,v in all_coefs.items():
        all_coefs[k] = float("{:.3f}".format(all_coefs[k]))
        all_of_all[k] += all_coefs[k]
    adf.at[sid, 'sample_id'] = sid
    adf.at[sid, 'avg_coef'] = avg_coef
    adf.at[sid, 'all_coefs'] = all_coefs
    all_of_avgs += avg_coef
    print('sid: ', sid, " avg_coef: ", avg_coef)

print([float("{:.3f}".format(v/len(sf))) for k,v in all_of_all.items()])
print(all_of_avgs/len(sf))
adf.to_csv('results/language-inference/lalor/SNLI_analysis_clustering-JSD_'+SET_ID+'.csv', index=False)
  