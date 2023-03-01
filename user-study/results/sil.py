import pandas as pd

#all_df = pd.read_csv('results/language-inference/lalor/SNLI_results_all-set.csv')
#user_study_df = pd.read_csv('user-study/results/filtered_results_Feb20.csv')

#for idx,row in user_study_df:
#    breakpoint()

all_set_df = pd.read_csv('results/language-inference/lalor/SNLI_results_all-set.csv')
user_study_results_df =  pd.read_csv('user-study/results/filtered_results_Feb20.csv')
us_df = pd.DataFrame({})
st =19
for idx, row in user_study_results_df.iterrows():
    user_study_results_df.columns[st]
    sent_1 = user_study_results_df.columns[st].split('\n')[0].split('Premise:')[1].strip()
    sent_2 = user_study_results_df.columns[st].split('\n\n\n')[1].split('Hypothesis:')[1].strip()
    sample_id = all_set_df[all_set_df.sentence_1 == sent_1].sample_id.values[0]
    us_df = us_df.append(all_set_df[all_set_df.sample_id == sample_id])
    
    #us_df.at[idx,'sentence_1'] = sent_1
    #us_df.at[idx,'sentence_2'] = sent_2
    st+=3

#us_df = us_df.sort_values(by=['sample_id'])
us_df.to_csv('SNLI_results_userstudy.csv', index=False)
