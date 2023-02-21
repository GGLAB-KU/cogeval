import pandas as pd
gold_file  = 'data/human/sentiment-analysis/lalor/sa_lalor_human.csv'
df = pd.read_csv(gold_file)

def get_label(sample_id):
    return int(df[df.sample_id ==sample_id].three_way_labels)


#def get_label(sst_phrase_id):
#    return int(df[df.sst_phrase_id ==sst_phrase_id].three_way_labels)


def get_machine_acc(file):
    mdf = pd.read_csv(file)
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        count+=1
        sid = row['sample_id']
        goldlabel = get_label(sid)
        predlabel = row['pred_label'] 
        if goldlabel == predlabel:
            true+=1
    
    return true/count




acc = get_machine_acc('data/machine/sentiment-analysis/lalor/davinci003-zeroshot/sentana_davinci-zeroshot_all.csv')
print("acc: %.3f" % acc)


