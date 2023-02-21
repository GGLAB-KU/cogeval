import pandas as pd

## Results file should include the following columns: 
# sample_id, 
# content, 
# gold_label,
# agg_human_label,
# agg_human_confidence,
# machine_<machineID>_pred,
# machine_<machineID>_confidence



def agree_on_trues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID)+'_pred'] == row['agg_human_label'] \
            and row['machine_'+str(machineID)+'_pred'] == row['gold_label'] :
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence']   = row['sentence']
                out_csv.at[i, 'domain']   = row['domain']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'agg_human_label'] = row['agg_human_label']
                out_csv.at[i, 'agg_human_confidence'] = row['agg_human_confidence']
                out_csv.at[i, 'machine_'+str(machineID)+'_pred'] = row['machine_'+str(machineID)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID)+'_confidence'] = row['machine_'+str(machineID)+'_confidence']
                conf_diff = abs(row['agg_human_confidence']-  row['machine_'+str(machineID)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
        #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv[['sample_id']] = out_csv[['sample_id']].astype(int)
        out_csv.to_csv(outdir +'_agree_on_trues.csv', index=False)
        print("# agree on trues: ", len(out_csv))

    else:
        print('NO such instance.')

def agree_on_falses(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID)+'_pred'] == row['agg_human_label'] \
            and row['machine_'+str(machineID)+'_pred'] != row['gold_label'] :
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence']   = row['sentence']
                out_csv.at[i, 'domain']   = row['domain']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'agg_human_label'] = row['agg_human_label']
                out_csv.at[i, 'agg_human_confidence'] = row['agg_human_confidence']
                out_csv.at[i, 'machine_'+str(machineID)+'_pred'] = row['machine_'+str(machineID)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID)+'_confidence'] = row['machine_'+str(machineID)+'_confidence']
                conf_diff = abs(row['agg_human_confidence']-  row['machine_'+str(machineID)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv[['sample_id']] = out_csv[['sample_id']].astype(int)
        out_csv.to_csv(outdir +'_agree_on_falses.csv', index=False)
        print("# agree on falses: ", len(out_csv))

    else:
        print('NO such instance.')

def disagree_on_humantrues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID)+'_pred'] != row['agg_human_label'] \
            and row['agg_human_label'] == row['gold_label']:
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence']   = row['sentence']
                out_csv.at[i, 'domain']   = row['domain']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'agg_human_label'] = row['agg_human_label']
                out_csv.at[i, 'agg_human_confidence'] = row['agg_human_confidence']
                out_csv.at[i, 'machine_'+str(machineID)+'_pred'] = row['machine_'+str(machineID)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID)+'_confidence'] = row['machine_'+str(machineID)+'_confidence']
                conf_diff = abs(row['agg_human_confidence'] + row['machine_'+str(machineID)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv[['sample_id']] = out_csv[['sample_id']].astype(int)
        out_csv.to_csv(outdir +'_disagree_on_humantrues.csv', index=False)
        print("# disagree on humantrues: ", len(out_csv))
    else:
        print('NO such instance.')

def disagree_on_machtrues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID)+'_pred'] != row['agg_human_label'] \
            and row['machine_'+str(machineID)+'_pred'] == row['gold_label']:
            out_csv.at[i, 'sample_id'] = row['sample_id']
            out_csv.at[i, 'sentence']   = row['sentence']
            out_csv.at[i, 'domain']   = row['domain']
            out_csv.at[i, 'gold_label'] = row['gold_label']
            out_csv.at[i, 'agg_human_label'] = row['agg_human_label']
            out_csv.at[i, 'agg_human_confidence'] = row['agg_human_confidence']
            out_csv.at[i, 'machine_'+str(machineID)+'_pred'] = row['machine_'+str(machineID)+'_pred']
            out_csv.at[i, 'machine_'+str(machineID)+'_confidence'] = row['machine_'+str(machineID)+'_confidence']
            conf_diff = abs(row['agg_human_confidence'] + row['machine_'+str(machineID)+'_confidence'])
            out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv[['sample_id']] = out_csv[['sample_id']].astype(int)
        out_csv.to_csv(outdir +'_disagree_on_machtrues.csv', index=False)
        print("# disagree on machtrues: ", len(out_csv))
    else:
        print('NO instance.')

def disagree_on_falses(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID)+'_pred'] != row['agg_human_label'] \
            and row['machine_'+str(machineID)+'_pred'] != row['gold_label'] \
            and row['agg_human_label'] != row['gold_label']:
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence']   = row['sentence']
                out_csv.at[i, 'domain']   = row['domain']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'agg_human_label'] = row['agg_human_label']
                out_csv.at[i, 'agg_human_confidence'] = row['agg_human_confidence']
                out_csv.at[i, 'machine_'+str(machineID)+'_pred'] = row['machine_'+str(machineID)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID)+'_confidence'] = row['machine_'+str(machineID)+'_confidence']
                conf_diff = abs(row['agg_human_confidence'] + row['machine_'+str(machineID)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
        #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv[['sample_id']] = out_csv[['sample_id']].astype(int)
        out_csv.to_csv(outdir +'_disagree_on_falses.csv', index=False)
    else:
        print('NO such instance.')


if __name__=="__main__":
    #TODO: generalize this
    #cols = ['sentence_1', 'sentence_2']
    results = pd.read_csv('results/linguistic-acceptability/CoLA_results_with_davinci003.csv')
    machineID = 1
    outdir = "results/linguistic-acceptability/human+machine"+str(machineID)+"/" + str(machineID)
    
    agree_on_trues()
    agree_on_falses()
    disagree_on_humantrues()
    disagree_on_machtrues()
    disagree_on_falses()
