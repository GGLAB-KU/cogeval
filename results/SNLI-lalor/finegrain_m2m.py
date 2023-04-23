import pandas as pd
import logging

## Results file should include the following columns: 
# sample_id, 
# sentence_1,
# sentence_2 
# gold_label,
# agg_human_label,
# agg_human_confidence,
# machine_<machineID>_pred,
# machine_<machineID>_confidence

def conf_scale(label, conf):
    if label == "neutral":
        conf = 0 - conf
    elif label == "contradiction":
        conf = -1 - conf
    return conf

def agree_on_trues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID_1)+'_pred'] == row['machine_'+str(machineID_2)+'_pred'] \
            and row['machine_'+str(machineID_1)+'_pred'] == row['gold_label'] :
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence_1']   = row['sentence_1']
                out_csv.at[i, 'sentence_2']   = row['sentence_2']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_pred'] = row['machine_'+str(machineID_1)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_confidence'] = row['machine_'+str(machineID_1)+'_confidence']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_pred'] = row['machine_'+str(machineID_2)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_confidence'] = row['machine_'+str(machineID_2)+'_confidence']
                
                conf_diff = abs(row['machine_'+str(machineID_2)+'_confidence']-  row['machine_'+str(machineID_1)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
        #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv.to_csv(outdir +'agree_on_trues_'+SET_ID+'.csv', index=False)
        logging.info("# agree on trues: %d" %len(out_csv))
    else:
        logging.info('NO such instance.')

def agree_on_falses(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID_1)+'_pred'] == row['machine_'+str(machineID_2)+'_pred'] \
            and row['machine_'+str(machineID_1)+'_pred'] != row['gold_label'] :
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence_1']   = row['sentence_1']
                out_csv.at[i, 'sentence_2']   = row['sentence_2']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_pred'] = row['machine_'+str(machineID_1)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_confidence'] = row['machine_'+str(machineID_1)+'_confidence']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_pred'] = row['machine_'+str(machineID_2)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_confidence'] = row['machine_'+str(machineID_2)+'_confidence']
                conf_diff = abs(row['machine_'+str(machineID_2)+'_confidence']-  row['machine_'+str(machineID_1)+'_confidence'])
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv.to_csv(outdir +'agree_on_falses_'+SET_ID+'.csv', index=False)
        logging.info("# agree on falses: %d" % len(out_csv))
    else:
        logging.info('NO such instance.')

def disagree_on_mach2trues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID_1)+'_pred'] != row['machine_'+str(machineID_2)+'_pred'] \
            and row['machine_'+str(machineID_2)+'_pred'] == row['gold_label']:
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence_1']   = row['sentence_1']
                out_csv.at[i, 'sentence_2']   = row['sentence_2']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_pred'] = row['machine_'+str(machineID_1)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_confidence'] = row['machine_'+str(machineID_1)+'_confidence']


                out_csv.at[i, 'machine_'+str(machineID_2)+'_pred'] = row['machine_'+str(machineID_2)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_confidence'] = row['machine_'+str(machineID_2)+'_confidence']


                mach2_conf = row['machine_'+str(machineID_2)+'_confidence']
                mach1_conf = row['machine_'+str(machineID_1)+'_confidence']
                conf_diff = abs(conf_scale( row['machine_'+str(machineID_2)+'_pred'],mach2_conf) - conf_scale(row['machine_'+str(machineID_1)+'_pred'],mach1_conf))
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv.to_csv(outdir +'disagree_on_mach2trues_'+SET_ID+'.csv', index=False)
        logging.info("# disagree on mach2trues: %d" % len(out_csv))
    else:
        logging.info('NO such instance.')

def disagree_on_mach1trues(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID_1)+'_pred'] !=row['machine_'+str(machineID_2)+'_pred']\
            and row['machine_'+str(machineID_1)+'_pred'] == row['gold_label']:
            out_csv.at[i, 'sample_id'] = row['sample_id']
            out_csv.at[i, 'sentence_1']   = row['sentence_1']
            out_csv.at[i, 'sentence_2']   = row['sentence_2']
            out_csv.at[i, 'gold_label'] = row['gold_label']
          
            out_csv.at[i, 'machine_'+str(machineID_1)+'_pred'] = row['machine_'+str(machineID_1)+'_pred']
            out_csv.at[i, 'machine_'+str(machineID_1)+'_confidence'] = row['machine_'+str(machineID_1)+'_confidence']
            out_csv.at[i, 'machine_'+str(machineID_2)+'_pred'] = row['machine_'+str(machineID_2)+'_pred']
            out_csv.at[i, 'machine_'+str(machineID_2)+'_confidence'] = row['machine_'+str(machineID_2)+'_confidence']


            mach2_conf = row['machine_'+str(machineID_2)+'_confidence']
            mach1_conf = row['machine_'+str(machineID_1)+'_confidence']
            conf_diff = abs(conf_scale( row['machine_'+str(machineID_2)+'_pred'],mach2_conf) - conf_scale(row['machine_'+str(machineID_1)+'_pred'],mach1_conf))
            out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
       #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv.to_csv(outdir +'disagree_on_mach1trues_'+SET_ID+'.csv', index=False)
        logging.info("# disagree on mach1trues: %d" % len(out_csv))
    else:
        logging.info('NO instance.')

def disagree_on_falses(ascending_diff=False):
    out_csv = pd.DataFrame({})
    for i, row in results.iterrows():
        if row['machine_'+str(machineID_1)+'_pred'] != row['machine_'+str(machineID_2)+'_pred'] \
            and row['machine_'+str(machineID_1)+'_pred'] != row['gold_label'] \
            and row['machine_'+str(machineID_2)+'_pred'] != row['gold_label']:
                out_csv.at[i, 'sample_id'] = row['sample_id']
                out_csv.at[i, 'sentence_1']   = row['sentence_1']
                out_csv.at[i, 'sentence_2']   = row['sentence_2']
                out_csv.at[i, 'gold_label'] = row['gold_label']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_pred'] = row['machine_'+str(machineID_1)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_1)+'_confidence'] = row['machine_'+str(machineID_1)+'_confidence']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_pred'] = row['machine_'+str(machineID_2)+'_pred']
                out_csv.at[i, 'machine_'+str(machineID_2)+'_confidence'] = row['machine_'+str(machineID_2)+'_confidence']
                
                mach2_conf = row['machine_'+str(machineID_2)+'_confidence']
                mach1_conf = row['machine_'+str(machineID_1)+'_confidence']
                conf_diff = abs(conf_scale( row['machine_'+str(machineID_2)+'_pred'],mach2_conf) - conf_scale(row['machine_'+str(machineID_1)+'_pred'],mach1_conf))
                out_csv.at[i, 'confidence_diff'] =conf_diff
    if len(out_csv) !=0:
        #out_csv[['machine_'+str(machineID)+'_pred']] = out_csv[['machine_'+str(machineID)+'_pred']].astype(int)
        #out_csv[['gold_label']] = out_csv[['gold_label']].astype(int)
        out_csv = out_csv.sort_values(by=['confidence_diff'], ascending=ascending_diff)
        out_csv.to_csv(outdir +'disagree_on_falses_'+SET_ID+'.csv', index=False)
        logging.info("# disagree on falses: %d " % len(out_csv))
    else:
        logging.info('NO such instance.')


if __name__=="__main__":
    
    SET_ID = "v3_fdf95_added"
    machineID_1 = 2
    machineID_2 = 3
    
    cols = ['sentence_1', 'sentence_2']
    results = pd.read_csv('results/SNLI-lalor/'+SET_ID+'.csv')

    outdir = "results/SNLI-lalor/"+ SET_ID+ "/machine"+str(machineID_1)+"_vs_machine"+str(machineID_2)+"/"
    logging.basicConfig(filename=outdir+'results.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    agree_on_trues()
    agree_on_falses()
    disagree_on_mach2trues()
    disagree_on_mach1trues()
    disagree_on_falses()