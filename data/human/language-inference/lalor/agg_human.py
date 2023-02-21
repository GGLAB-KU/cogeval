import pandas as pd
import numpy as np

rawhuman = pd.read_csv('data/human/language-inference/lalor/snli_human_4gs_raw.csv')

out_df = pd.DataFrame({'snli_id': [], 'pred_label': []})

true = 0
for idx, row in rawhuman.iterrows():
    agg_human_pred = row[1:].value_counts().idxmax()
    out_df.at[idx,'snli_id'] = row['snli_id']
    if agg_human_pred == 'c':
        out_df.at[idx, 'pred_label'] = 'contradiction'
    if agg_human_pred == 'e':
        out_df.at[idx, 'pred_label'] = 'entailment'
    if agg_human_pred == 'n':
        out_df.at[idx, 'pred_label'] = 'neutral'
    agg_human_conf_1 = np.array(row[1:].value_counts())[0]/ sum(np.array(row[1:].value_counts()))
    agg_human_conf_2 = np.array(row[1:].value_counts())[1]/ sum(np.array(row[1:].value_counts()))
    agg_human_conf_3 = np.array(row[1:].value_counts())[2]/ sum(np.array(row[1:].value_counts()))
    out_df.at[idx, row[1:].value_counts().keys()[:3][0]] = agg_human_conf_1
    out_df.at[idx, row[1:].value_counts().keys()[:3][1]] = agg_human_conf_2
    out_df.at[idx, row[1:].value_counts().keys()[:3][2]] = agg_human_conf_3
    agg_human_conf = np.array(row[1:].value_counts())[0]/ sum(np.array(row[1:].value_counts()))
    out_df.at[idx, 'confidence'] = agg_human_conf 
    out_df.at[idx, 'sample_id'] = row['sample_id']
    out_df[['sample_id']] = out_df[['sample_id']].astype(int)

out_df.set_index(out_df.columns[-1], inplace=True)
out_df.to_csv('agg_hum_snli_distribution.csv')
    
    