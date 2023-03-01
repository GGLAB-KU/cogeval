import pandas as pd
gold_file  = 'data/human/language-inference/lalor/snli_human_4gs.csv'
df = pd.read_csv(gold_file)

def get_label(sample_id):
    return df[df.sample_id ==sample_id].label


#def get_label(sst_phrase_id):
#    return int(df[df.sst_phrase_id ==sst_phrase_id].three_way_labels)


def get_machine_acc(file):
    mdf = pd.read_csv(file)
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        predlabel = row['pred_label']
        sid = row['sample_id']
        if isinstance(predlabel, str):
            count+=1
            goldlabel = get_label(sid)
            goldlabel = goldlabel.values[0].strip()
            if goldlabel == predlabel.strip():
                true+=1
            else:
                print(goldlabel, predlabel)
    
    print('# instances:', count)
    return true/count


def get_machine_acc_on_subset(file, subsetfile):
    mdf = pd.read_csv(file)
    sdf = pd.read_csv(subsetfile)
    
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        predlabel = row['pred_label']
        sid = row['sample_id']
        if int(sid) in list(sdf['sample_id']):
            #print(sid)
            if isinstance(predlabel, str):
                count+=1
                goldlabel = get_label(sid)
                goldlabel = goldlabel.values[0].strip()
                if goldlabel == predlabel.strip():
                    true+=1
                else:
                    pass
                    #print(goldlabel, predlabel)
        
    print('# instances:', count)
    return true/count

models = ['random', 'tfidf', 'lstm', 'roberta', 'davinci', 'human']
subset_file = 'user-study/questions/userstudy.csv'
for model in models:
    acc = get_machine_acc_on_subset('results/language-inference/lalor/files/'+model+'.csv', subset_file)
    print('model:', model, " acc: %.3f" % acc)


