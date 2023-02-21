import pandas as pd
import random

machines =[]
machine_0 = pd.read_csv('data/machine/linguistic-acceptability/random/cola_random.csv')
machine_1 = pd.read_csv('data/machine/linguistic-acceptability/RoBERTa/cola_roberta.csv') # RoBERTa 
machine_2 = pd.read_csv('data/machine/linguistic-acceptability/DeBERTa/cola_deberta.csv') # DeBERTa 
machine_3 = pd.read_csv('data/machine/linguistic-acceptability/davinci003-twoshot/cola_davinci-twoshot_all.csv') # DeBERTa 
agg_human_pred = pd.read_csv("data/human/linguistic-acceptability/agg_hum_cola.csv") # Human agg.

machines = [machine_0, machine_1, machine_2, machine_3]
human = "data/human/linguistic-acceptability/human_judgments.csv"
human = pd.read_csv(human)

out_df =  pd.DataFrame({
'sample_id': [], 
'sentence' :  [],
'gold_label': [],
'machine_0_pred': [], 
'machine_0_confidence': [], 
'machine_1_pred': [], 
'machine_1_confidence': [], 
'machine_2_pred': [], 
'machine_2_confidence': [], 
'machine_3_pred': [], 
'machine_3_confidence': [], 
'agg_human_label': [],
'agg_human_confidence': []
})

humantrue = 0
total = 0
for i, hrow in human.iterrows():
    total+=1
    out_df.at[i, 'sentence'] = hrow['sentence']
    out_df.at[i, 'domain'] = hrow['domain']

    out_df.at[i, 'gold_label'] = hrow['CoLA_label']
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
out_df.to_csv('CoLA_results_with_davinci003.csv', index=False)