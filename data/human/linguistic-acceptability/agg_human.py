import pandas as pd
import numpy as np

rawhuman = pd.read_csv('data/human/linguistic-acceptability/human_judgments.csv')
out_df = pd.DataFrame({'sample_id': [], 'agg_human_conf': [], 'agg_human_pred': [], 'gold_label': []})

for idx, row in rawhuman.iterrows():
    agg_human_pred = row[5:-1].value_counts().idxmax()
    agg_human_conf = np.array(row[5:-1].value_counts())[0]/ sum(np.array(row[5:-1].value_counts()))
   
    out_df.at[idx,'sample_id'] = row['sample_id']
    out_df.at[idx, 'agg_human_pred'] = agg_human_pred
    out_df.at[idx, 'agg_human_conf'] = agg_human_conf
    out_df.at[idx, 'gold_label'] = row['CoLA_label']

out_df[['sample_id']] = out_df[['sample_id']].astype(int)
out_df[['agg_human_pred']] = out_df[['agg_human_pred']].astype(int)
out_df[['gold_label']] = out_df[['gold_label']].astype(int)
out_df.to_csv('agg_hum_cola.csv', index=False)
    
    