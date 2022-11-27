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
    agg_human_conf = np.array(row[1:].value_counts())[0]/ sum(np.array(row[1:].value_counts()))
    out_df.at[idx, 'agg_human_conf'] = agg_human_conf
    out_df.at[idx, 'sample_id'] = row['sample_id']
    out_df[['sample_id']] = out_df[['sample_id']].astype(int)

out_df.set_index(out_df.columns[-1], inplace=True)
out_df.to_csv('agg_hum_snli.csv')
    
    