import openai, time
import pandas as pd
import numpy as np
#muge
#openai.api_key = "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" 
#sadra
openai.api_key = "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" 

df = pd.read_csv("data/human/linguistic-acceptability/human_judgments.csv")
out_df = pd.DataFrame({'sample_id': [], 'sentence': [], 'pred_label': [], 'gold_label': []})


for i,row in df.iterrows():
    if row['sample_id'] <= 50:
        continue
    print(row['sample_id'])

    sentence = row['sentence']
    prompt_text = "Sentence: She mailed John a letter, but I don't know to whom.\nThe sentence is not grammatical.\n\nSentence: The newspaper has reported that they are about to appoint someone, but I can't remember who the newspaper has reported that they are about to appoint.\nThe sentence is grammatical.\n\nSentence: " + sentence + "\nIs the sentence grammatical? Answer as yes or no.\n\nAnswer:"

    print(prompt_text)
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
    out_df.at[i, 'sentence'] = row['sentence']
    out_df.at[i, 'gold_label'] = row['CoLA_label']

    pred_label =  response.choices[0].text.strip()
  
    if pred_label == 'yes' or pred_label == 'Yes':
        out_df.at[i, 'pred_label'] = 1
    elif pred_label == 'no' or pred_label == 'No':
        out_df.at[i, 'pred_label'] = 0
    else:
        print("problem with prompt:" , prompt_text, "->", pred_label)

    if row['sample_id'] %25 == 0 and row['sample_id']!=0:
        out_df[['sample_id']]  = out_df[['sample_id']].astype(int)
        out_df.to_csv('cola_davinci-zeroshot_'+str(row['sample_id'])+'.csv', index=False)
        time.sleep(60)

out_df.to_csv('cola_davinci-zeroshot_all.csv', index=False)
