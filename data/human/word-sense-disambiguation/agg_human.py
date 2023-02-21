import pandas as pd
import numpy as np

rawhuman = pd.read_csv('data/human/word-sense-disambiguation/ann3_4_wic.csv')

out_df = pd.DataFrame({'sample_id': [], 'pred_label': [], 'gold_label': [], 'agg_human_conf': []})

true = 0
disag = 0
for idx, row in rawhuman.iterrows():
    if row[1] != row[2]:
        disag+=1
        print(disag)
    agg_human_pred = row[1:3].value_counts().idxmax()
    out_df.at[idx,'sample_id'] = row['sample_id']
    if agg_human_pred == 'T':
        out_df.at[idx, 'pred_label'] = 1
    if agg_human_pred == 'F':
        out_df.at[idx, 'pred_label'] = 0
    agg_human_conf = np.array(row[1:3].value_counts())[0]/ sum(np.array(row[1:3].value_counts()))
    out_df.at[idx, 'agg_human_conf'] = agg_human_conf
    out_df.at[idx, 'sample_id'] = row['sample_id']
    out_df.at[idx, 'sentence_1'] =  row['sentence_1']
    out_df.at[idx, 'sentence_2'] =  row['sentence_2']
    out_df.at[idx, 'word'] =  row['word']

    
    if row['gold_label'] == 'T':
        out_df.at[idx, 'gold_label'] = 1
    else:
        out_df.at[idx, 'gold_label'] = 0

    out_df[['sample_id']] = out_df[['sample_id']].astype(int)


out_df.to_csv('agg_hum_wic.csv', index=False)
    
    