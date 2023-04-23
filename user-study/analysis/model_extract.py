import pandas as pd
import numpy as np
from collections import defaultdict

def get_model(df, pred, matched):
    diffs = []
    mach0 = df['machine_0_confidence'].values[0]
    mach1 = df['machine_1_confidence'].values[0]
    mach2 = df['machine_2_confidence'].values[0]
    mach3 = df['machine_3_confidence'].values[0]
    mach4 = df['machine_4_confidence'].values[0]
    lalorhuman = df['agg_human_confidence'].values[0]

    if 0 in matched:
        diffs.append(abs(pred-mach0))
    else:   
        diffs.append(9999)
    
    if 1 in matched:
        diffs.append(abs(pred-mach1))
    else:
        diffs.append(9999)


    if 2 in matched:
        diffs.append(abs(pred-mach2))
    else:
        diffs.append(9999)


    if 3 in matched:
        diffs.append(abs(pred-mach3))
    else:
        diffs.append(9999)


    if 4 in matched:
        diffs.append(abs(pred-mach4))
    else:
        diffs.append(9999)
    
    if 5 in matched:
        diffs.append(abs(pred-lalorhuman))
    else:
        diffs.append(9999)      

    modelID = np.argmin(diffs)
    return modelID

def get_models_with_same_label(df, label):
    matched = []
    if df['machine_0_pred'].values[0] == label:
        matched.append(0)

    if df['machine_1_pred'].values[0] == label:
        matched.append(1)

    if df['machine_2_pred'].values[0] == label:
        matched.append(2)

    if df['machine_3_pred'].values[0] == label:
        matched.append(3)

    if df['machine_4_pred'].values[0] == label:
        matched.append(4)
  
    if df['agg_human_label'].values[0] == label:
        matched.append(5)
    return matched


nonfiltered_us_results_df = pd.read_csv('user-study/results/nonfiltered_results_March7.csv')
filtered_us_results_df =  pd.read_csv('user-study/results/filtered_results_March7.csv')
all_set_df = pd.read_csv('results/SNLI-lalor/merged_results_all.csv')
meta_df = pd.DataFrame({})

USER_ID = 'R_3G2sevJRBdumSIc'

st =31
while True:
    NUM_USER = 0
    correct_count = 0
    cs = 0
    ns = 0
    es = 0
    cs_probs = 0
    ns_probs = 0
    es_probs = 0
    
    selectedmodels = defaultdict(lambda:0)
    if st > 180:
        break
    for idx, row in filtered_us_results_df.iterrows():
        response_id = row['Response ID']
        if response_id != USER_ID:
            continue
        NUM_USER+=1
        #per user
        participant_answ = row[filtered_us_results_df.columns[st]]
        participant_answ = participant_answ.replace('>','')
        participant_answ = participant_answ.replace('<','')
        contradiction, neutral, entailment= participant_answ.strip().split("|")
        contradiction = contradiction.strip()
        neutral = neutral.strip()
        entailment = entailment.strip()
        contr = float(contradiction.split(' ')[0])
        neut = float(neutral.split(' ')[0])
        ent = float(entailment.split(' ')[0])
        cs_probs+=contr
        ns_probs+=neut
        es_probs+=ent
        labelID= np.argmax([contr,neut,ent])
        if labelID == 0:
            label = 'contradiction'
            cs +=1
        elif labelID == 1:
            label = 'neutral'
            ns +=1
        elif labelID == 2:
            label = 'entailment'
            es +=1

        sent_1 = nonfiltered_us_results_df.values[0][st-1].split('\n')[0].split('Premise:')[1].strip()
        sent_2 = nonfiltered_us_results_df.values[0][st-1].split('\n\n\n')[1].split('Hypothesis:')[1].strip()
        print(sent_2)
        
        sample_id = all_set_df[all_set_df.sentence_2 == sent_2].sample_id.values[0]
        #if sample_id == 57:
        #    breakpoint()
        smodels = get_models_with_same_label(all_set_df[all_set_df.sample_id==sample_id], label)
        selectedmodel = get_model(all_set_df[all_set_df.sample_id==sample_id], np.max([contr,neut,ent]), smodels)
      

        selectedmodels[selectedmodel] +=1
        #for sm in smodels: change for all models
        #    selectedmodels[sm] +=1
        gold_label = all_set_df[all_set_df.sample_id==sample_id]['gold_label'].values[0]
        if gold_label == label:
            correct_count +=1
    
        #break

    ## end of all users results
    meta_df.at[st, 'response_id'] = response_id
    meta_df.at[st, 'sample_id'] = sample_id
    meta_df.at[st, 'selectedmodel'] = selectedmodel

    labelID= np.argmax([cs,ns,es])
    if labelID == 0:
        agg_label = 'contradiction'
    elif labelID == 1:
        agg_label = 'neutral'
    elif labelID == 2:
        agg_label = 'entailment'

    meta_df.at[st, 'participant_answ'] = participant_answ
    meta_df.at[st, 'avg_userstudy_accuracy'] = (correct_count/NUM_USER)

    #meta_df.at[st, 'avg_userstudy_confidence'] = max([cs_probs,ns_probs,es_probs])/NUM_USER
    #meta_df.at[st, 'avg_userstudy_c'] = (cs_probs/NUM_USER)
    #meta_df.at[st, 'avg_userstudy_n'] = (ns_probs/NUM_USER)
    #meta_df.at[st, 'avg_userstudy_e'] = (es_probs/NUM_USER)

    meta_df.at[st, 'agg_userstudy_accuracy'] = (gold_label == agg_label)
    #meta_df.at[st, 'agg_userstudy_confidence'] =  max(ns,cs,es)/NUM_USER
    meta_df.at[st, 'agg_userstudy_label'] = agg_label
    meta_df.at[st, 'gold_label']   = all_set_df[all_set_df.sentence_2 == sent_2].gold_label.values[0]
    
    #meta_df.at[st, 'c'] =  cs/NUM_USER
    #meta_df.at[st, 'n'] =  ns/NUM_USER
    #meta_df.at[st, 'e'] =  es/NUM_USER

    for i in range(6):
        meta_df.at[st, i] =  selectedmodels[i]
    st+=3
meta_df[['sample_id']] = meta_df[['sample_id']].astype(int)
meta_df.to_csv('user-study/results/user_results_'+USER_ID+'.csv', index=False)
