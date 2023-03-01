import pandas as pd
import numpy as np
import math
from collections import defaultdict

df = pd.read_csv("user-study/results/filtered_results_Feb20.csv")
dif_cols = [col for col in df.columns if 'difficult' in col]


i=1
qid = 1
answers = dict()
diffs = dict()
while i< len(dif_cols):
    mylist = df[dif_cols[i]]
    answers[qid] = [0 if math.isnan(x) else x for x in mylist]
    diffs[qid] = sum(answers[qid])
    i+=2
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
df = pd.read_csv("results/language-inference/lalor/SNLI_results_userstudy.csv")
for idx,row in df.iterrows():
    row['diff_score'] = diffs_list[idx+1]
    row['diff_rank']  = diff_ranks[idx+1]
    ndf = ndf.append(row)
ndf.to_csv("user-study/results/results_with_diff_scores.csv", index=False)

