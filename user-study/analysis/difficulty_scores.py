import pandas as pd
import numpy as np
import math
from collections import defaultdict

df = pd.read_csv("user-study/results/filtered_results_March7.csv")
dif_cols = ['Q'+str(idx) for idx in range(459,510)]

i=1
qid = 1
answers = dict()
diffs = dict()
while i< len(dif_cols):
    print(dif_cols[i])
    mylist = df[dif_cols[i]]
    answers[qid] = [0 if math.isnan(x) else x for x in mylist]
    total = np.nansum([x for x in mylist])
    num_of_notnan = 50-sum(np.isnan([x for x in mylist]))
    diffs[qid] = total/ num_of_notnan
    i+=1
    qid+=1
diffs = {k: v for k, v in sorted(diffs.items(), key=lambda item: item[1])}

diffs_list = dict()
diff_ranks = dict()
d = 50
for k,v in diffs.items():
    diff_ranks[k] = d
    diffs_list[k] = v
    d-=1

ndf = pd.DataFrame({})
avg_df = pd.read_csv("user-study/results/results_with_meta.csv")

for idx,row in avg_df.iterrows():
    row['diff_score'] = diffs_list[idx+1]
    row['diff_rank']  = diff_ranks[idx+1]
    ndf = ndf.append(row)
ndf.to_csv("user-study/results/results_with_diff_scores.csv", index=False)

