import pandas as pd
import random

machines =[]
machine_0 = pd.read_csv('data/machine/language-inference/lalor/random/snli_random.csv')
machine_1 = pd.read_csv('data/machine/language-inference/lalor/TF-IDF/snli_tfidf.csv') # TF-IDF + Logistic Regression
machine_2 = pd.read_csv('data/machine/language-inference/lalor/LSTM/snli_lstm_calibrated.csv') # LSTM
machine_3 = pd.read_csv('data/machine/language-inference/lalor/RoBERTa/snli_lalor_roberta.csv') # roberta
agg_human_pred = pd.read_csv('data/human/language-inference/lalor/agg_hum_snli.csv') # Human agg.

machines = [machine_0, machine_1, machine_2, machine_3]
human = "data/human/language-inference/lalor/snli_human_4gs.csv"
human = pd.read_csv(human)


out_df =  pd.DataFrame({
'sample_id': [], 
'sentence_1' :  [],
'sentence_2' :  [],
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
    out_df.at[i, 'sentence_1'] = hrow['sentence_1']
    out_df.at[i, 'sentence_2'] = hrow['sentence_2']
    out_df.at[i, 'gold_label'] = hrow['label']
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
out_df.to_csv('SNLI_results.csv', index=False)