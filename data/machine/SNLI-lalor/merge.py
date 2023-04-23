import pandas as pd
import random

SET_ID = "v3_all"

machines =[]
machine_0 = pd.read_csv('data/machine/SNLI-lalor/'+SET_ID+'/random_noisy.csv')
machine_1 = pd.read_csv('data/machine/SNLI-lalor/'+SET_ID+'/tfidf.csv') # TF-IDF + Logistic Regression
machine_2 = pd.read_csv('data/machine/SNLI-lalor/'+SET_ID+'/lstm.csv') # LSTM
machine_3 = pd.read_csv('data/machine/SNLI-lalor/'+SET_ID+'/roberta_calibrated.csv') # roberta
machine_4 = pd.read_csv('data/machine/SNLI-lalor/'+SET_ID+'/davinci.csv') # davinci
#machine_5 = pd.read_csv('data/machine/SNLI-lalor/v2_tilek/lstm.csv') # davinci
#machine_6 = pd.read_csv('data/machine/SNLI-lalor/v2_tilek/roberta.csv') # davinci
agg_human_pred = pd.read_csv('data/human/SNLI-lalor/agg_hum_snli.csv') # Human agg.
#agg_userstudy_pred = pd.read_csv('user-study/results/results_with_diff_scores.csv') # Userstudy human agg.

machines = [machine_0, machine_1, machine_2, machine_3, machine_4]#, machine_5, machine_6]
human = "data/human/SNLI-lalor/snli_human_4gs.csv"
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
'machine_4_pred': [], 
'machine_4_confidence': [], 
#'machine_5_pred': [], 
#'machine_5_confidence': [], 
#'machine_6_pred': [], 
#'machine_6_confidence': [], 
'agg_human_label': [],
'agg_human_confidence': [],
#'agg_userstudy_label': [],
#'agg_userstudy_confidence': []
})

#subsetdf = pd.read_csv('results/language-inference/lalor/SNLI_results_LSTM_ROBERTA_HUMAN_AGREE.csv')
#subset_ids = (subsetdf.sample_id).tolist()

humantrue = 0
total = 0
for i, hrow in human.iterrows():
    #if hrow['sample_id'] in subset_ids:
    total+=1
    out_df.at[i, 'sentence_1'] = hrow['sentence_1'].strip()
    out_df.at[i, 'sentence_2'] = hrow['sentence_2'].strip()
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

    #agg_userstudy_row = agg_userstudy_pred.loc[agg_userstudy_pred['sample_id'] == hrow['sample_id']]
    #out_df.at[i, 'agg_userstudy_label'] = agg_userstudy_row['agg_userstudy_label'].values[0]
    #out_df.at[i, 'agg_userstudy_confidence'] = agg_userstudy_row['agg_userstudy_conf']

print('#total:', total)
out_df.to_csv('results/SNLI-lalor/'+SET_ID+'.csv', index=False)