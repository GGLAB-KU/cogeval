import openai, time
import pandas as pd
import numpy as np
#muge
#openai.api_key = "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" 
#sadra
openai.api_key = "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" 

df = pd.read_csv("data/human/word-sense-disambiguation/ann3_4_wic.csv")
out_df = pd.DataFrame({'sample_id': [], 'word': [], 'sentence_1': [], 'sentence_2': [], 'pred_label': [], 'gold_label': []})


for i,row in df.iterrows():
    print(row['sample_id'])

    sentence_1 = row['sentence_1']
    sentence_2 = row['sentence_2']
    word = row['word']

    prompt_text="Sentence: " + sentence_1 + "\n\n" + "Another sentence: " + sentence_2 + "\n\n" + "Does the word \"" + word + "\"" + " have the same lexical meaning in these sentences? Answer as yes or no. Explain your reasoning." + "\n\n" + "Answer:"
    out_df.at[i, 'prompt_text'] = prompt_text

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_text,
        temperature=0.0,
        max_tokens=1,
        top_p=1,
        logprobs= 3,
        frequency_penalty=0,
        presence_penalty=0
    )
    confidence =  np.exp(response['choices'][0]['logprobs']['token_logprobs'][0])
    out_df.at[i, 'confidence'] = confidence
    out_df.at[i, 'logprobs'] = str(response['choices'][0]['logprobs']['top_logprobs'][0])
    out_df.at[i, 'sample_id'] = row['sample_id']
    out_df.at[i, 'word'] = row['word']
    out_df.at[i, 'sentence_1'] = row['sentence_1']
    out_df.at[i, 'sentence_2'] = row['sentence_2']
    
    if row['gold_label'] == 'T':
        out_df.at[i, 'gold_label'] = 1
    else:
        out_df.at[i, 'gold_label'] = 0

    pred_label =  response.choices[0].text.strip()
  
    if pred_label == 'yes' or pred_label == 'Yes':
        out_df.at[i, 'pred_label'] = 1
    elif pred_label == 'no' or pred_label == 'No':
        out_df.at[i, 'pred_label'] = 0
    else:
        print("problem with prompt:" , prompt_text, "->", pred_label)

    if row['sample_id'] %25 == 0 and row['sample_id']!=0:
        out_df[['sample_id']]  = out_df[['sample_id']].astype(int)
        out_df.to_csv('wic_davinci-zeroshot_'+str(row['sample_id'])+'.csv', index=False)
        time.sleep(60)

out_df.to_csv('wic_davinci-zeroshot_all_reasoning.csv', index=False)
