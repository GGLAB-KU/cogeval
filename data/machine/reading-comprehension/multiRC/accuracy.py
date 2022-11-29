import pandas as pd

def get_machine_acc(file):
    mdf = pd.read_csv(file)
    count = 0
    true = 0
    for idx, row in mdf.iterrows():
        count+=1
        goldlabel = row['gold_label'] 
        predlabel = row['pred_label'] 
        if goldlabel == predlabel:
            true+=1
    acc = true/count
    return acc

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

def get_human_f1_a(file):
    mdf = pd.read_csv(file)
    correct_count = 0
    agreement_count = 0
    predict_count = 0
    for idx, row in mdf.iterrows():
        if row['agg_human_pred'] == 1 or row['agg_human_conf'] == 0.5:
            predict_count+=1
        goldlabel = row['gold_label'] 
        if goldlabel == 1:
            correct_count+=1
            if row['agg_human_pred'] == goldlabel or row['agg_human_conf'] == 0.5:
                agreement_count+=1
    recall = agreement_count/correct_count
    precision   = agreement_count/predict_count
    f1_a = 2 * (precision * recall) / (precision + recall)
    return precision, recall,f1_a

def get_machine_f1_a(file):
    mdf = pd.read_csv(file)
    correct_count = 0
    agreement_count = 0
    predict_count = 0
    for idx, row in mdf.iterrows():
        goldlabel = row['gold_label'] 
        if row['pred_label'] == 1: #or row['pred_conf'] == 0.5:
            predict_count+=1
        if goldlabel == 1:
            correct_count+=1
            if row['pred_label'] == goldlabel:# or row['pred_conf'] == 0.5:
                agreement_count+=1
    #print('correct count' , correct_count)
    #print('agreement count' , agreement_count)
    #print('predict count' , predict_count)
    recall = agreement_count/correct_count
    precision   = agreement_count/predict_count
    f1_a = 2 * (precision * recall) / (precision + recall)
    return precision, recall,f1_a


machinefile = 'data/machine/reading-comprehension/multiRC/random/multirc_random.csv'

print('---')
print('Machine statistics:')
precision, recall, f1_a = get_machine_f1_a(machinefile)
acc = get_machine_acc(machinefile)
print('precision: %.3f, recall: %.3f, f1_a: %.3f' %(precision, recall,f1_a))
print('acc: %.3f' % acc)

'''print('---')
print('Human statistics:')
precision, recall, f1_a = get_human_f1_a(machinefile)
acc = get_human_acc(machinefile)
print('precision: %.3f, recall: %.3f, f1_a: %.3f' %(precision, recall,f1_a))
print('acc: %.3f' % acc)'''


