import openai, time
import pandas as pd
import numpy as np
from collections import defaultdict

openai.api_key = <OPEN_AI_KEY> 

def calc_dist(logprobs):
    dists = defaultdict(lambda:0)
    for k,v in logprobs.items():
        ()
        if k.strip().lower() == 'maybe':
            dists['neutral'] +=np.exp(v)
        if k.strip().lower() == 'no':
            dists['contradiction'] += np.exp(v)
        if k.strip().lower() == 'yes':
            dists['entailment'] +=np.exp(v)

    if len(dists) <2:
        if 'neutral' not in dists:
            dists['neutral'] = (1-sum(dists.values()))/2
        if 'contradiction' not in dists:
          dists['contradiction'] = (1-sum(dists.values()))/2
        if 'entailment' not in dists:
          dists['entailment'] = (1-sum(dists.values()))/2
    
    if 'neutral' not in dists:
        dists['neutral'] = 1-sum(dists.values())
    elif 'contradiction' not in dists:
        dists['contradiction'] = 1-sum(dists.values())
    elif 'entailment' not in dists:
        dists['entailment'] = 1-sum(dists.values())

    return dists


df = pd.read_csv("data/human/language-inference/lalor/snli_human_4gs.csv")
out_df = pd.DataFrame({'sample_id': [], 'sentence_1': [], 'sentence_2': [], 'pred_label': [], 'gold_label': []})



for i,row in df.iterrows():
    if row['sample_id'] <= 50:
        continue
    print(row['sample_id'])
    sent_1 = row['sentence_1']
    #sent_1 = sent_1[0].lower() + sent_1[1:]
    sent_2 = row['sentence_2']
    sent_2 = sent_2[0].lower() + sent_2[1:]
    if sent_2[-1] == '.':
        sent_2 = sent_2[:-1]
    prompt_text = sent_1 + "\n\n" + "Can we infer that " + sent_2 + "?" + " Answer as yes, no or maybe.\n\n" + "Answer:"
    print(prompt_text)
    out_df.at[i, 'prompt_text'] = prompt_text

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text,
        temperature=0.0,
        max_tokens=1,
        top_p=1,
        logprobs= 5,
        frequency_penalty=0,
        presence_penalty=0
    )
    confidence =  np.exp(response['choices'][0]['logprobs']['token_logprobs'][0])
    dists = calc_dist(response['choices'][0]['logprobs']['top_logprobs'][0])
    out_df.at[i, 'confidence'] = confidence
    out_df.at[i, 'logprobs'] = str(response['choices'][0]['logprobs']['top_logprobs'][0])
    out_df.at[i, 'sample_id'] = row['sample_id']
    out_df.at[i, 'sentence_1'] = row['sentence_1']
    out_df.at[i, 'sentence_2'] = row['sentence_2']
    out_df.at[i, 'gold_label'] = row['label']

    pred_label =  response.choices[0].text.strip()
    print(pred_label)
    print(dists)
    print('-----------------')
    out_df.at[i, 'c'] = dists['contradiction']
    out_df.at[i, 'n'] = dists['neutral']
    out_df.at[i, 'e'] = dists['entailment']


    if pred_label == 'yes' or pred_label == 'Yes':
        out_df.at[i, 'pred_label'] = 'entailment'
    elif pred_label == 'maybe' or pred_label == 'Maybe':
        out_df.at[i, 'pred_label'] = 'neutral'
    elif pred_label == 'No' or pred_label == 'no':
        out_df.at[i, 'pred_label'] = 'contradiction'
    else:
        print("problem with prompt:" , prompt_text, "->", pred_label)

    if row['sample_id'] %25 == 0 and row['sample_id']!=0:
        out_df[['sample_id']]  = out_df[['sample_id']].astype(int)
        out_df.to_csv('snli_davinci-zeroshot_'+str(row['sample_id'])+'_.csv', index=False)
        time.sleep(60)

out_df.to_csv('snli_davinci-results.csv', index=False)




        
