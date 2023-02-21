import pandas as pd
import random

machines =[]
machine_0 = pd.read_csv('data/machine/reading-comprehension/multiRC/random/multirc_random.csv')
machine_1 = pd.read_csv('data/machine/reading-comprehension/multiRC/RoBERTa/with_meta.csv') # RoBERTa + human data
machine_2 = pd.read_csv('data/machine/reading-comprehension/multiRC/davinci-003-zeroshot/multirc_davinci-zeroshot_0-300.csv') # text-davinci-003

machines = [machine_0, machine_1, machine_2]

human = "data/machine/reading-comprehension/multiRC/RoBERTa/with_meta.csv"
human = pd.read_csv(human)


out_df =  pd.DataFrame({
'sample_id': [], 
'paragraph' :  [],
'question' :  [],
'answer-option' :  [],
'gold_label': [],
'machine_0_pred': [], 
'machine_0_confidence': [], 
'machine_1_pred': [], 
'machine_1_confidence': [], 
'machine_2_pred': [], 
'machine_2_confidence': [], 
'agg_human_label': [],
'agg_human_confidence': []
})

humantrue = 0
total = 0
for i, hrow in human.iterrows():
    total+=1
    out_df.at[i, 'paragraph'] = hrow['paragraph']
    out_df.at[i, 'question'] = hrow['question']
    out_df.at[i, 'gold_label'] = hrow['gold_label']
    out_df.at[i, 'answer-option'] = hrow['answer-option']
    out_df.at[i, 'sample_id'] = hrow['sample_id']
    out_df.at[i, 'agg_human_label'] = hrow['agg_human_pred']
    
    if hrow['agg_human_score'] == 0 or hrow['agg_human_score'] == 1:
        out_df.at[i, 'agg_human_confidence'] = 1
    elif hrow['agg_human_score'] == 0.25 or hrow['agg_human_score'] == 0.75:
        out_df.at[i, 'agg_human_confidence'] = 0.75
    elif hrow['agg_human_score'] == 0.50:
        out_df.at[i, 'agg_human_confidence'] = 0.50

    out_df[['sample_id']] = out_df[['sample_id']].astype(int)
    for j, machine in enumerate(machines):
        mrow = machine.loc[machine['sample_id'] == hrow['sample_id']]
        if len(mrow)==0:
            continue
        out_df.at[i, 'machine_'+str(j)+'_pred'] = mrow['pred_label'].values[0]
        out_df.at[i, 'machine_'+str(j)+'_confidence'] = mrow['confidence']
  

print('#total:', total)
out_df.to_csv('MultiRC_results_with_davinci003.csv', index=False)