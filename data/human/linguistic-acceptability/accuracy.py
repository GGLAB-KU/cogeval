import pandas as pd
import math

def get_human_acc(file):
    mdf = pd.read_csv(file)
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        count+=1
        goldlabel = row['gold_label'] 
        predlabel = row['agg_human_pred'] 
        if goldlabel == predlabel:
            true+=1
    acc = true/count
    return acc

def get_human_mcc(file):
    tp, fp, fn, tn = 0, 0, 0, 0
    mdf = pd.read_csv(file)
    for idx, row in mdf.iterrows():
        goldlabel = row['gold_label'] 
        predlabel = row['agg_human_pred'] 
        if goldlabel == predlabel == 1:
            tp +=1
        elif goldlabel == predlabel == 0: 
            tn +=1
        elif (goldlabel != predlabel)  and (predlabel == 1):
            fp +=1
        elif goldlabel != predlabel == 0:
            fn +=1
    try:
        mcc = float((tp * tn) - (fp * fn)) / \
                                            math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    except ZeroDivisionError:
        mcc = 0
    return mcc



humanfile = 'data/human/linguistic-acceptability/agg_hum_cola.csv'
print('---')
print('Human statistics:')
acc = get_human_acc(humanfile)
print('acc: %.3f' % acc)
mcc = get_human_mcc(humanfile)
print('mcc: %.3f' % mcc)

