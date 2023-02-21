import pandas as pd
import random

machines =[]
machine_0 = pd.read_csv('data/machine/word-sense-disambiguation/random/wic_random.csv')
machine_1 = pd.read_csv('data/machine/word-sense-disambiguation/davinci003-zeroshot/wic_davinci-zeroshot_all_reasoning.csv') # davinci-003-zeroshot 
agg_human_pred = pd.read_csv("data/human/word-sense-disambiguation/agg_hum_wic.csv") # Human agg.

machines = [machine_0, machine_1]
human = "data/human/word-sense-disambiguation/agg_hum_wic.csv"
human = pd.read_csv(human)

out_df =  pd.DataFrame({
'sample_id': [], 
'sentence' :  [],
'gold_label': [],
'machine_0_pred': [], 
'machine_0_confidence': [], 
'machine_1_pred': [], 
'machine_1_confidence': [], 
'agg_human_label': [],
'agg_human_confidence': []
})

humantrue = 0
total = 0
for i, hrow in human.iterrows():
    total+=1
    out_df.at[i, 'sentence_1'] = hrow['sentence_1']
    out_df.at[i, 'sentence_2'] = hrow['sentence_2']
    out_df.at[i, 'word'] = hrow['word']
    out_df.at[i, 'gold_label'] = hrow['gold_label']
    out_df.at[i, 'sample_id'] = hrow['sample_id']
    out_df[['sample_id']] = out_df[['sample_id']].astype(int)
    
    for j, machine in enumerate(machines):
        mrow = machine.loc[machine['sample_id'] == hrow['sample_id']]
        out_df.at[i, 'machine_'+str(j)+'_pred'] = mrow['pred_label'].values[0]
        out_df.at[i, 'machine_'+str(j)+'_confidence'] = mrow['confidence']
    aggrow = agg_human_pred.loc[agg_human_pred['sample_id'] == hrow['sample_id']]
    out_df.at[i, 'agg_human_label'] = aggrow['pred_label'].values[0]
    out_df.at[i, 'agg_human_confidence'] = aggrow['agg_human_conf']

print('#total:', total)
out_df.to_csv('wic_results_with_davinci003_with_reasoning.csv', index=False)