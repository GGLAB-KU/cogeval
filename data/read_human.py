import csv
import pandas as pd

def read_sentiment(fname):
    '''data = []
    with open(fname) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for row in rdr:
            if row[0]=='sstb_id':
                keys = row
            else:
                instance = {}
                for key,value in zip(keys,row):
                    instance[key] = value
                data.append(instance)
    return data'''
    # easier to use dataframe
    dataframe = pd.read_csv(fname)
    return dataframe

def read_re(fname):
    data = []
    with open(fname) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for row in rdr:
            if row[0]!='paragraph_id':
                data.append({'par_id': row[0],'sent_id': row[1], 'sent':row[2], 'rel':row[3],
                             'label':row[4], 'fix':row[5], 'trt':row[9], 'omission':row[13],
                             'pupil':row[17], 'accuracy': row[30], 'IRT_p1': row[31], 'IRT_p4': row[32]})
    return data

def read_qa(fname):
    data = []
    with open(fname) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for row in rdr:
            if row[0]!='paragraph_id':
                data.append({'par_id': row[0],'sent_id': row[1], 'sent':row[2], 'rel':row[3],
                             'label':row[4], 'fix':row[5], 'trt':row[9], 'omission':row[13],
                             'pupil':row[17], 'accuracy': row[30], 'IRT_p1': row[31], 'IRT_p4': row[32]})
    return data

def read_snli(fname):
    data = []
    with open(fname) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for row in rdr:
            if row[0]!='paragraph_id':
                data.append({'par_id': row[0],'sent_id': row[1], 'sent':row[2], 'rel':row[3],
                             'label':row[4], 'fix':row[5], 'trt':row[9], 'omission':row[13],
                             'pupil':row[17], 'accuracy': row[30], 'IRT_p1': row[31], 'IRT_p4': row[32]})
    return data

def get_human_data(file_name, task):
    data = []
    if task == "sentiment":
        data = read_sentiment(file_name)
    elif task == "re":
        data = read_re(file_name)
    elif task == "qa":
        data = read_qa(file_name)
    elif task == "snli":
        data = read_snli(file_name)
    return data

