import pandas as pd
from itertools import combinations


def calculate_inst_affect(inst_results, result, mach):
    new_total_acc = inst_results+[result]
    new_acc_score = sum(new_total_acc)/len(new_total_acc)
    instance_affect = abs(overall_accuracies[mach] - new_acc_score)
    #print('instance_affect:', instance_affect, ' on ', mach)
    return new_total_acc, instance_affect

def one_model_inst_affect(keys):
    inst_results = []
    tot = 0
    for k in keys:
        v = meta_instance_accs[k]
        inst_results, instance_affect = calculate_inst_affect(inst_results, v['m0'],'m0')
        tot += instance_affect
        #print('on instance: ', k , '-->', instance_affect)
    print('total: ', tot)

def all_models_inst_affect(keys):
    inst_effects = {
        'm0':[], 
        'm1':[],
        'm2':[],
        'm3':[],
        'm4':[],
        'm5':[]}
    for mach in ['m0', 'm1', 'm2', 'm3','m4','m5']:
        all_models_instance_affect = 0
        inst_results = []
        for k in keys:
            v = meta_instance_accs[k]
            inst_results, instance_affect = calculate_inst_affect(inst_results, v[mach], mach)
            inst_effects[mach].append(instance_affect)
        #print('instance:', k, ' allmodel_instance_affect:', all_models_instance_affect)
    return inst_effects

def nodes_for_comb(comb):
    inst_effects = all_models_inst_affect(comb)
    nodes = []
    for em0,em1,em2,em3,em4,em5 in zip(inst_effects['m0'], inst_effects['m1'], inst_effects['m2'], inst_effects['m3'], inst_effects['m4'], inst_effects['m5']):
        total_effect = em0 + em1 + em2 + em3 + em4 + em5
        nodes.append(total_effect)
    return min(nodes)

def get_best_subset(set,N):
    combs = list(combinations(set, N))
    res = dict()
    for idx,comb in enumerate(combs):
        res[idx] = nodes_for_comb(comb)
    temp = min(res.values())
    MIN_COMB_IDX = [key for key in res if res[key] == temp][0]
    print('N:', N, ' len_combs:', len(combs), ' len(set):', len(set))
    return list(combs[MIN_COMB_IDX]), temp


df = pd.read_csv('results/language-inference/lalor/SNLI_analysis_true-false_all-set.csv')
overall_accuracies = {'m0': 0, 'm1': 0, 'm2':0, 'm3':0, 'm4':0, 'm5':0}
meta_instance_accs = dict()
for idx, row in df.iterrows():
    instance_accuracies = {'m0': 0, 'm1': 0, 'm2':0, 'm3':0, 'm4':0, 'm5':0}
    tm = row['true_machs']
    true_machs = []
    try:
        if tm[1:-1] != '':
            for j in tm[1:-1].replace(' ','').split(','):
                true_machs.append(int(j))
                overall_accuracies['m'+j] +=1
                instance_accuracies['m'+j] +=1
    except:
        pass
    meta_instance_accs[idx] = instance_accuracies


for k,v in overall_accuracies.items():
    overall_accuracies[k] /= 90 


lm = [*range(0,90,1)]
lm.reverse()
i = 89
while i>=50:
    lm, temp = get_best_subset(lm,i)
    i = i-1
out_df = pd.DataFrame({'sample_id':[]})
for sid in lm:
    out_df.loc[len(out_df.index)] = sid

out_df.to_csv('user-study/questions/greedy.csv')


'''lm = [*range(0,90,1)]
blm = [*range(0,90,1)]
lm.reverse()
c1 = list(combinations(blm, 89))
c2 = list(combinations(lm,89))

a1, t1 = get_best_subset(blm,89)
a2, t2 = get_best_subset(lm,89)
breakpoint()'''

#lm = [*range(0,90,1)]
#lm.reverse()
#one_model_inst_affect(lm)


'''
### Find best subset for a model
lm = [*range(0,90,1)]
combs = list(combinations(lm, 50))
all_accs = []
for comb in combs:
    acc = 0
    for inst in comb:
        acc += meta_instance_accs[inst]['m0']
    all_accs.append(abs(overall_accuracies['m0']- (acc/len(comb))))
breakpoint()'''
