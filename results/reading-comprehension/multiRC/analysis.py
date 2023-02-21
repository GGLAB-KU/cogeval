from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
def save_heatmap(corr):
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='RdBu_r')
    plt.xticks(rotation = 60)
    plt.savefig('multirc_analysis.png', dpi=600, bbox_inches='tight')


results = pd.read_csv('results/reading-comprehension/multiRC/MultiRC_results_with_davinci003.csv')
analysis_df = pd.DataFrame({'sample_id':[],'paragraph':[], 'question':[], 'answer-option':[],'gold_label':[], 'agg_human_label':[], 'agg_human_confidence':[], 'agg_mach_label':[], 'agg_mach_confidence':[] })
num_machines = 3

for idx, row in results.iterrows():
    if idx>300:
        break
    num_trues = 0
    num_falses =0
    true_machs = []
    false_machs = []
    goldlabel = row['gold_label']
    analysis_df.at[idx, 'sample_id'] = row['sample_id']

    analysis_df.at[idx, 'paragraph']   = row['paragraph']
    analysis_df.at[idx, 'question']   = row['question']
    analysis_df.at[idx, 'answer-option']   = row['answer-option']

    analysis_df.at[idx, 'gold_label']    = goldlabel
    analysis_df.at[idx, 'agg_human_label']   = row['agg_human_label']
    analysis_df.at[idx, 'agg_human_confidence']   = row['agg_human_confidence']
   
    mach_preds = [row[i] for i in [5,7,9]]
    cnt = Counter(mach_preds)
    aggpred =cnt.most_common(1)
    agg_mach_label = int(aggpred[0][0])

    for mi in range(num_machines):
        mpred = row['machine_'+str(mi)+'_pred']
        if goldlabel == mpred:
            num_trues+=1
            true_machs.append(mi)
        else:
            num_falses +=1
            false_machs.append(mi)

    analysis_df.at[idx, 'num_trues']    = num_trues
    analysis_df.at[idx, 'num_falses']   = num_falses
    analysis_df.at[idx, 'false_machs']  = str(false_machs)
    analysis_df.at[idx, 'true_machs']   = str(true_machs)
    analysis_df.at[idx, 'agg_mach_label']   = agg_mach_label
    analysis_df.at[idx, 'agg_mach_confidence']   = aggpred[0][1]/num_machines

corr = analysis_df[['agg_human_confidence', 'agg_mach_confidence']].corr()
heatmap      = save_heatmap(corr)


analysis_df.to_csv('multirc_analysis.csv', index=False)


        

