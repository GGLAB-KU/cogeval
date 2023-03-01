import pandas as pd

df = pd.read_csv('results/language-inference/lalor/SNLI_analysis_true-false_subset1.csv')

ALL_TRUE = 0
ROBERTA_AND_LSTM_TRUE = 0
LSTM_AND_HUMAN_TRUE = 0
ROBERTA_AND_HUMAN_TRUE = 0

for idx, row in df.iterrows():
    true_machs = row['true_machs']
    if ('2' in true_machs) and ('3' in true_machs) and ('5' in true_machs):
        ALL_TRUE +=1
    elif ('2' in true_machs) and ('3' in true_machs) and ('5' not in true_machs):
        ROBERTA_AND_LSTM_TRUE +=1
    elif ('2' in true_machs) and ('3' not in true_machs) and ('5' in true_machs):
        LSTM_AND_HUMAN_TRUE +=1
    elif ('2' not in true_machs) and ('3' in true_machs) and ('5' in true_machs):
        ROBERTA_AND_HUMAN_TRUE +=1

print('ALL_TRUE: ', ALL_TRUE)
print('ROBERTA_AND_LSTM_TRUE: ', ROBERTA_AND_LSTM_TRUE)
print('LSTM_AND_HUMAN_TRUE: ', LSTM_AND_HUMAN_TRUE)
print('ROBERTA_AND_HUMAN_TRUE: ', ROBERTA_AND_HUMAN_TRUE)

## NO! Change this!

ALL_FALSE = 0
ROBERTA_AND_LSTM_FALSE = 0
LSTM_AND_HUMAN_FALSE = 0
ROBERTA_AND_HUMAN_FALSE = 0
ONLY_ROBERTA_FALSE = 0
ONLY_HUMAN_FALSE = 0

human_df = pd.read_csv('results/language-inference/lalor/files/human.csv')
roberta_df = pd.read_csv('results/language-inference/lalor/files/roberta.csv')
lstm_df = pd.read_csv('results/language-inference/lalor/files/lstm.csv')

for idx, row in df.iterrows():
    false_machs = row['false_machs']
    sample_id = int(row['sample_id'])

    human_pred =  (human_df[human_df.sample_id==sample_id].pred_label).values[0].strip()
    roberta_pred =  (roberta_df[roberta_df.sample_id==sample_id].pred_label).values[0].strip()
    lstm_pred =  (lstm_df[lstm_df.sample_id==sample_id].pred_label).values[0].strip()
    if ('2' in false_machs) and ('3' in false_machs) and ('5' in false_machs) and  (roberta_pred == lstm_pred) and (lstm_pred == human_pred):
        ALL_FALSE +=1
        print('ALL_FALSE:', sample_id)
    if ('2' in false_machs) and ('3' in false_machs) and ('5' in false_machs) and  (roberta_pred == lstm_pred) and (lstm_pred != human_pred):
        ROBERTA_AND_LSTM_FALSE +=1
    if ('2' in false_machs) and ('3' in false_machs) and ('5' not in false_machs) and (roberta_pred == lstm_pred):
        ROBERTA_AND_LSTM_FALSE +=1
    if ('2' in false_machs) and ('3' not in false_machs) and ('5' in false_machs) and (human_pred == lstm_pred):
        LSTM_AND_HUMAN_FALSE +=1
        #print('LSTM_AND_HUMAN_FALSE:', sample_id)
    if ('2' not in false_machs) and ('3' in false_machs) and ('5' in false_machs) and (roberta_pred == human_pred):
        ROBERTA_AND_HUMAN_FALSE +=1
        print('ROBERTA_AND_HUMAN_FALSE:', sample_id)
    if ('3' in false_machs) and ('2' not in false_machs) and ('5' not in false_machs):
        ONLY_ROBERTA_FALSE +=1
    if ('5' in false_machs): #and ('2' not in false_machs) and ('3' not in false_machs):
        print('ONLY_HUMAN_FALSE:', sample_id)
        ONLY_HUMAN_FALSE +=1

print('------------------')
#print('ALL_FALSE: ', ALL_FALSE)
#print('ROBERTA_AND_LSTM_FALSE: ', ROBERTA_AND_LSTM_FALSE)
print('LSTM_AND_HUMAN_FALSE: ', LSTM_AND_HUMAN_FALSE)
print('ROBERTA_AND_HUMAN_FALSE: ', ROBERTA_AND_HUMAN_FALSE)
print('ONLY_HUMAN_FALSE: ', ONLY_HUMAN_FALSE)