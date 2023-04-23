import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_csv("user-study/results/filtered_results_March7.csv")

roberta_df = pd.DataFrame({})
human_df = pd.DataFrame({})
davinci_df = pd.DataFrame({})

alignments = defaultdict(lambda:0)
instead_of_you_answers = defaultdict(lambda:0)
coop_answers = defaultdict(lambda:0)

for idx, row in df.iterrows():
    model0_score = row['model0_score']
    model1_score = row['model1_score']
    model2_score = row['model2_score']
    model3_score = row['model3_score']
    model4_score = row['model4_score']
    human_score  = row['human_score']
    scores = [model0_score, model1_score, model2_score, model3_score, model4_score, human_score]
    aligned_agent_ID = np.argmax(scores)
    alignments[aligned_agent_ID] +=1


    j = 0
    for key in row.keys():
        if 'Would you prefer to use' in key:
            if j == aligned_agent_ID:
                instead_of_you = row[key]
            j+=1


    j = 0
    for key in row.keys():
        if 'Would you cooperate' in key:
            if j == aligned_agent_ID:
                cooperation = row[key]
            j+=1
    
    j = 0
    for key in row.keys():
        if 'Please briefly explain' in key:
            if j == aligned_agent_ID:
                opinions = row[key]
            j+=1


    j = 0
    for key in row.keys():
        if 'Do you think the agent is a human?' in key:
            if j == aligned_agent_ID:
                it_is_a_human = row[key]
            j+=1

    j = 0
    for key in row.keys():
        if 'Do you think the agent is a machine?' in key:
            if j == aligned_agent_ID:
                it_is_a_machine = row[key]
            j+=1
    
    task_difficulty = row['How difficult was the task?']
    task_fun = row['How fun was the task?']
    task_agency_help = row['How helpful would it be to use this agent for this task?']
    task_subjective = row['How subjective is the task?']
    task_ability_logical_thinking = row['How important are the following abilities for solving the task? - Logical thinking']
    task_ability_emotional_awareness = row['How important are the following abilities for solving the task? - Emotional awareness']
    task_ability_native_speaker = row['How important are the following abilities for solving the task? - Being a native English speaker']
    user_accuracy = row['user_accuracy']/50

    instead_of_you_answers[instead_of_you]+=1
    coop_answers[cooperation]+=1
    response_ID =  row['Response ID']

    if aligned_agent_ID == 3:
        roberta_df.at[idx, 'response_ID'] = response_ID
        roberta_df.at[idx, 'model0_score'] = model0_score
        roberta_df.at[idx, 'model1_score'] = model1_score
        roberta_df.at[idx, 'model2_score'] = model2_score
        roberta_df.at[idx, 'model3_score'] = model3_score
        roberta_df.at[idx, 'model4_score'] = model4_score
        roberta_df.at[idx, 'human_score'] = human_score

        roberta_df.at[idx, 'instead_of_you'] = instead_of_you
        roberta_df.at[idx, 'cooperation'] = cooperation
        roberta_df.at[idx, 'it_is_a_human'] = it_is_a_human
        roberta_df.at[idx, 'it_is_a_machine'] = it_is_a_machine
        roberta_df.at[idx, 'opinions'] = opinions
        roberta_df.at[idx, 'user_accuracy'] = user_accuracy

        #roberta_df.at[idx, 'task_difficulty'] = task_difficulty

    if aligned_agent_ID == 4:
        davinci_df.at[idx, 'response_ID'] = response_ID
        davinci_df.at[idx, 'model0_score'] = model0_score
        davinci_df.at[idx, 'model1_score'] = model1_score
        davinci_df.at[idx, 'model2_score'] = model2_score
        davinci_df.at[idx, 'model3_score'] = model3_score
        davinci_df.at[idx, 'model4_score'] = model4_score
        davinci_df.at[idx, 'human_score'] = human_score

        davinci_df.at[idx, 'instead_of_you'] = instead_of_you
        davinci_df.at[idx, 'cooperation'] = cooperation
        davinci_df.at[idx, 'it_is_a_human'] = it_is_a_human
        davinci_df.at[idx, 'it_is_a_machine'] = it_is_a_machine
        davinci_df.at[idx, 'opinions'] = opinions
        davinci_df.at[idx, 'user_accuracy'] = user_accuracy

    if aligned_agent_ID == 5:
        human_df.at[idx, 'response_ID'] = response_ID
        human_df.at[idx, 'model0_score'] = model0_score
        human_df.at[idx, 'model1_score'] = model1_score
        human_df.at[idx, 'model2_score'] = model2_score
        human_df.at[idx, 'model3_score'] = model3_score
        human_df.at[idx, 'model4_score'] = model4_score
        human_df.at[idx, 'human_score'] = human_score

        human_df.at[idx, 'instead_of_you'] = instead_of_you
        human_df.at[idx, 'cooperation'] = cooperation
        human_df.at[idx, 'it_is_a_human'] = it_is_a_human
        human_df.at[idx, 'it_is_a_machine'] = it_is_a_machine
        human_df.at[idx, 'opinions'] = opinions
        human_df.at[idx, 'user_accuracy'] = user_accuracy


roberta_df.to_csv('user-study/results/aligned_robertas.csv', index=False)
davinci_df.to_csv('user-study/results/aligned_davincis.csv', index=False)
human_df.to_csv('user-study/results/aligned_humans.csv', index=False)

print("Instead of you: ")
print(instead_of_you_answers)
print("Coop answers: ")
print(coop_answers)
