from collections import Counter
import pandas as pd
import numpy as np
from data.read_human import *
import random

def read_data(fname):
    # easier to use dataframe
    dataframe = pd.read_csv(fname)
    return dataframe

def stamp_sentiment_analysis_data(machinefile, humanfile, outputfile):
    machine_df = pd.read_csv(machinefile)
    human_df = pd.read_csv(humanfile)
    for i, row in machine_df.iterrows():
        sample_id = int((human_df[human_df.sst_phrase_id == row.sst_phrase_id]).sample_id) 
        #machine_df.at[i, 'sample_id'] = sample_id
        machine_df.at[i, 'pred_label'] = row['pred_label'] - 1
    machine_df[['sample_id']] = machine_df[['sample_id']].astype(int)
    #machine_df.set_index(machine_df.columns[-1], inplace=True)
    machine_df.to_csv(outputfile, index=False)

def stamp_multirc_data(machinefile):
    df = pd.read_csv(machinefile)
    for i, row in df.iterrows():
        df.at[i, 'sample_id']= i
    df[['sample_id']] = df[['sample_id']].astype(int)
    df.set_index(df.columns[-1], inplace=True)
    df.to_csv(machinefile)

def stamp_cola_data(machinefile):
    df = pd.read_csv(machinefile, delimiter=',')
    for i, row in df.iterrows():
        df.at[i, 'sample_id']= i
    df[['sample_id']] = df[['sample_id']].astype(int)
    df.set_index(df.columns[-1], inplace=True)
    df.to_csv(machinefile)

def stamp_snli_data(machinefile, humanfile):
    humandf = pd.read_csv(humanfile)
    df = pd.read_csv(machinefile)
    for i, row in df.iterrows():
        sid = humandf[humandf.snli_id==row['snli_id']].sample_id
        df.at[i, 'sample_id']= int(sid)
    df[['sample_id']] = df[['sample_id']].astype(int)
    df.set_index(df.columns[-1], inplace=True)
    df.to_csv(machinefile)


def stamp_wic_data(machinefile, humanfile):
    humandf = pd.read_csv(humanfile)
    df = pd.read_csv(machinefile)
    for i, row in humandf.iterrows():
        df.at[i, 'sample_id']= row['sample_id']
    df[['sample_id']] = df[['sample_id']].astype(int)
    df.set_index(df.columns[-1], inplace=True)
    df.to_csv(machinefile)




def randomize_multirc(file):
    df = pd.read_csv(file)
    random_df = pd.DataFrame({'sample_id':[], 'confidence':[], 'pred_label':[], 'gold_label': []})
    for i,row in df.iterrows():
        random_df.at[i, 'sample_id'] = row['sample_id']
        random_df.at[i, 'confidence'] = 1/2
        ri = random.randint(0,1)
        random_df.at[i, 'pred_label'] = ri
        random_df.at[i, 'gold_label'] = row['gold_label']
    random_df[['sample_id']] = random_df[['sample_id']].astype(int)
    random_df[['pred_label']] = random_df[['pred_label']].astype(int)
    random_df[['gold_label']] = random_df[['gold_label']].astype(int)
    random_df.to_csv('multirc_random.csv',index=False)


def randomize_cola(file):
    df = pd.read_csv(file)
    random_df = pd.DataFrame({'sample_id':[], 'confidence':[], 'pred_label':[], 'CoLA_label': []})
    for i,row in df.iterrows():
        random_df.at[i, 'sample_id'] = row['sample_id']
        random_df.at[i, 'confidence'] = 1/2
        ri = random.randint(0,1)
        random_df.at[i, 'pred_label'] = ri
        random_df.at[i, 'CoLA_label'] = row['CoLA_label']

    random_df[['sample_id']] = random_df[['sample_id']].astype(int)
    random_df[['pred_label']] = random_df[['pred_label']].astype(int)
    random_df[['CoLA_label']] = random_df[['CoLA_label']].astype(int)
    random_df.to_csv('cola_random.csv',index=False)


def randomize_wic(file):
    df = pd.read_csv(file)
    random_df = pd.DataFrame({'sample_id':[], 'confidence':[], 'pred_label':[], 'gold_label': []})
    for i,row in df.iterrows():
        random_df.at[i, 'sample_id'] = row['sample_id']
        random_df.at[i, 'confidence'] = 1/2
        ri = random.randint(0,1)
        random_df.at[i, 'pred_label'] = ri
        if row['gold_label'] == 'T':
            random_df.at[i, 'gold_label'] = 1
        else:
            random_df.at[i, 'gold_label'] = 0

    random_df[['sample_id']] = random_df[['sample_id']].astype(int)
    #random_df[['pred_label']] = random_df[['pred_label']].astype(int)
    #random_df[['gold_label']] = random_df[['gold_label']].astype(int)
    random_df.to_csv('wic_random.csv',index=False)

def randomize_snli(file):
    df = pd.read_csv(file)
    random_df = pd.DataFrame({'sample_id':[], 'confidence':[], 'pred_label':[]})
    for i,row in df.iterrows():
        random_df.at[i, 'sample_id'] = row['sample_id']
        noise =  np.random.normal(0, 0.05, 3)
        ps = [1/3, 1/3, 1/3]
        ps += noise
        ps /=ps.sum(); 
        random_df.at[i, 'confidence'] = max(ps)
        random_df.at[i, 'c'] = ps[0]
        random_df.at[i, 'n'] = ps[1]
        random_df.at[i, 'e'] = ps[2]

        ri = np.argmax(ps) -1
        if ri == -1:
            random_df.at[i, 'pred_label'] = 'contradiction'
        if ri == 0:
            random_df.at[i, 'pred_label'] = 'neutral'
        if ri == 1:
            random_df.at[i, 'pred_label'] = 'entailment'
    random_df[['sample_id']] = random_df[['sample_id']].astype(int)
    random_df.to_csv('random_noisy.csv',index=False)





if __name__ == '__main__':
    #stamp_cola_data('data/machine/linguistic-acceptability/DeBERTa/cola_deberta.csv')
    #stamp_snli_data('user-study/lstm.csv', 'data/human/language-inference/lalor/snli_human_4gs.csv')
    #randomize_wic('data/human/word-sense-disambiguation/ann3_4_wic.csv')
    #randomize_multirc('data/machine/reading-comprehension/multiRC/RoBERTa/multirc_roberta_calib.csv')
    randomize_snli('data/human/SNLI-lalor/snli_human_4gs.csv')