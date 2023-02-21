import pandas as pd
import random

machines =[]
machine_0 = pd.read_csv('data/machine/sentiment-analysis/lalor/random/random.csv')
machine_1 = pd.read_csv('data/machine/sentiment-analysis/lalor/TF-IDF/all.csv') # TF-IDF + Logistic Regression
machine_2 = pd.read_csv('data/machine/sentiment-analysis/lalor/BERT/mc_dropout/all.csv') # BERT
machine_3 = pd.read_csv('data/machine/sentiment-analysis/lalor/davinci003-zeroshot/sentana_davinci-zeroshot_all.csv') # davinci003-zeroshot
machine_4 = pd.read_csv('data/machine/sentiment-analysis/lalor/RoBERTa/sa_lalor_human_roberta.csv') # roberta
machine_5 = pd.read_csv('data/machine/sentiment-analysis/lalor/openGPT-finetuned/classifier.csv') # openGPT+ trained classifier

agg_human_pred = pd.read_csv('data/human/sentiment-analysis/lalor/agg_hum.csv') # Human agg.

machines = [machine_0, machine_1, machine_2, machine_3, machine_4, machine_5]
human = "data/human/sentiment-analysis/lalor/sa_lalor_human.csv"
human = pd.read_csv(human)


out_df =  pd.DataFrame({
'sample_id': [], 
'content' :  [],
'gold_label': [],
'machine_0_pred': [], 
'machine_0_confidence': [], 
'machine_1_pred': [], 
'machine_1_confidence': [], 
'machine_2_pred': [], 
'machine_2_confidence': [], 
'machine_3_pred': [], 
'machine_3_confidence': [], 
'machine_4_pred': [], 
'machine_4_confidence': [], 
'machine_5_pred': [], 
'machine_5_confidence': [], 
'agg_human_label': [],
'agg_human_confidence': []
})

humantrue = 0
total = 0
for i, hrow in human.iterrows():
    total+=1
    out_df.at[i, 'content'] = hrow['content']
    out_df.at[i, 'gold_label'] = hrow['three_way_labels']
    out_df.at[i, 'sample_id'] = hrow['sample_id']
    out_df[['sample_id']] = out_df[['sample_id']].astype(int)
    for j, machine in enumerate(machines):
        mrow = machine.loc[machine['sample_id'] == hrow['sample_id']]
        out_df.at[i, 'machine_'+str(j)+'_pred'] = mrow['pred_label']
        out_df.at[i, 'machine_'+str(j)+'_confidence'] = mrow['confidence']
    aggrow = agg_human_pred.loc[agg_human_pred['sst_id'] == hrow['sst_phrase_id']]
    out_df.at[i, 'agg_human_label'] = aggrow['pred_label']
    out_df.at[i, 'agg_human_confidence'] = aggrow['agg_human_conf']

print('#total:', total)
out_df.to_csv('SST_results_v2.csv', index=False)
