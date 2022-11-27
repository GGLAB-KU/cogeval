import openai, time
import pandas as pd

openai.api_key = "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" 

df = pd.read_csv("data/human/language-inference/lalor/snli_human_4gs.csv")

out_df = pd.DataFrame({'sample_id': [], 'sentence_1': [], 'sentence_2': [], 'pred_label': []})
for i,row in df.iterrows():
    print(i)
    if i %25 == 0:
        time.sleep(60)
    
    sent_1 = row['sentence_1']
    sent_2 = row['sentence_2']
   
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Decide whether the inference between two sentences is contradiction, neutral, or entailment.\n\n" + \
            "Sentence 1: Two men and a woman are inspecting the front tire of a bicycle.\n"     + \
            "Sentence 2: There are a group of people near a bike. \nInterence: entailment.\n\n" + \
            "Decide whether the inference between two sentences is contradiction, neutral, or entailment.\n\n" + \
            "Sentence 1: "+ sent_1 + ".\n" + \
            "Sentence 2: "+ sent_2 + "\nInference: \n\n",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    pred_label =  response.choices[0].text.strip()
    out_df.at[i, 'sample_id'] = row['sample_id']
    out_df.at[i, 'sentence_1'] = row['sentence_1']
    out_df.at[i, 'sentence_2'] = row['sentence_2']

    if pred_label == 'contradiction' or pred_label == 'Contradiction':
        out_df.at[i, 'pred_label'] = 'contradiction'
    elif pred_label == 'neutral' or pred_label == 'Neutral':
        out_df.at[i, 'pred_label'] = 'neutral'
    elif pred_label == 'entailment' or pred_label == 'Entailment':
        out_df.at[i, 'pred_label'] = 'entailment'
    else:
        print("problem with sentences: " , sent_1 ," ", sent_2, " ->", pred_label)

out_df[['sample_id']] = out_df[['sample_id']].astype(int)
out_df.to_csv('prompt_out_snli.csv', index=False)