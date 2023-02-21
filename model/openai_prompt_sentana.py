import openai, time
import pandas as pd
import numpy as np
#muge
#openai.api_key = "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" 
#sadra
openai.api_key = "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" 

df = pd.read_csv("data/human/sentiment-analysis/lalor/sa_lalor_human_preprocessed.csv")
out_df = pd.DataFrame({'sample_id': [], 'content': [], 'pred_label': [], 'gold_label': []})


for i,row in df.iterrows():
    #if row['sample_id'] < 75:
    #    continue
    print(row['sample_id'])

    review = row['content']
    prompt_text = review + "\n\n" + "What can we say about the sentiment of this review? Answer as positive, negative or neutral.\n\nAnswer:"

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
    breakpoint()
    out_df.at[i, 'confidence'] = confidence
    out_df.at[i, 'logprobs'] = str(response['choices'][0]['logprobs']['top_logprobs'][0])
    out_df.at[i, 'sample_id'] = row['sample_id']
    out_df.at[i, 'content'] = row['content']
    out_df.at[i, 'gold_label'] = row['three_way_labels']

    pred_label =  response.choices[0].text.strip()
  
    if pred_label == 'positive' or pred_label == 'Positive':
        out_df.at[i, 'pred_label'] = 1
    elif pred_label == 'neutral' or pred_label == 'Neutral':
        out_df.at[i, 'pred_label'] = 0
    elif pred_label == 'negative' or pred_label == 'Negative':
        out_df.at[i, 'pred_label'] = -1
    else:
        print("problem with prompt:" , prompt_text, "->", pred_label)

    if row['sample_id'] %25 == 0 and row['sample_id']!=0:
        out_df[['sample_id']]  = out_df[['sample_id']].astype(int)
        out_df.to_csv('sentana_davinci-zeroshot_'+str(row['sample_id'])+'.csv', index=False)
        time.sleep(60)

out_df.to_csv('sentana_davinci-zeroshot_all.csv', index=False)
