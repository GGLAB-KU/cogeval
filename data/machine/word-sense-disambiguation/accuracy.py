import pandas as pd
import math

def get_machine_acc(file):
    mdf = pd.read_csv(file)
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        count+=1
        goldlabel = row['gold_label'] 
        if goldlabel == 'T':
            goldlabel = 1
        elif goldlabel == 'F':
            goldlabel = 0
        predlabel = row['pred_label'] 
        if goldlabel == predlabel:
            true+=1
    acc = true/count
    print('#instances: ', count)
    return acc

machinefile = 'data/machine/word-sense-disambiguation/davinci003-zeroshot/wic_davinci-zeroshot_all_reasoning.csv'
print('---')
print('Machine statistics:')
acc = get_machine_acc(machinefile)
print('acc: %.3f' % acc)
