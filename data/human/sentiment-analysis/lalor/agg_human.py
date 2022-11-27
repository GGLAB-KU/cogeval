import pandas as pd
import numpy as np

rawhuman = pd.read_csv('/kuacc/users/mugekural/workfolder/dev/git/cogeval/data/human/sentiment-analysis/lalor/sa_lalor_human_raw.csv')

out_df = pd.DataFrame({'sst_id': [], 'pred_label': []})

true = 0
for idx, row in rawhuman.iterrows():
    agg_human_pred = row[1:].value_counts().idxmax()
    out_df.at[idx,'sst_id'] = row['sst_id']
    if agg_human_pred == 1 or agg_human_pred == 2:
        out_df.at[idx, 'pred_label'] = -1
    elif agg_human_pred == 4 or agg_human_pred == 5:
        out_df.at[idx, 'pred_label'] = 1
    elif agg_human_pred == 3:
        out_df.at[idx, 'pred_label'] = 0
    agg_human_conf = np.array(row[1:].value_counts())[0]/ sum(np.array(row[1:].value_counts()))
    out_df.at[idx, 'agg_human_conf'] = agg_human_conf

    
out_df[['pred_label']] = out_df[['pred_label']].astype(int)
out_df[['sst_id']] = out_df[['sst_id']].astype(int)

out_df.to_csv('agg_hum2.csv', index=False)
    
    