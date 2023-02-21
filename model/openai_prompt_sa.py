import openai, time
import pandas as pd
openai.api_key = "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" 

df = pd.read_csv("data/human/sentiment-analysis/lalor/sa_lalor_human_preprocessed.csv")
out_df = pd.DataFrame({'sample_id': [], 'content': [], 'pred_label': []})

for i,row in df.iterrows():
    if i %25 == 0:
        time.sleep(60)
    content = row['content']
    review = content
    #review = "I loved the new Batman movie!"
    review = "\""+ review+"\"" + "\n"

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Decide whether a Tweet's sentiment is positive, neutral, or negative.\n\Tweet:" + review +"Sentiment:",
    temperature=0,
    max_tokens=10,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0
    )

    pred_label =  response.choices[0].text.strip()
    out_df.at[i, 'sample_id'] = row['sample_id']
    out_df.at[i, 'content'] = row['content']

    if pred_label == 'positive' or pred_label == 'Positive':
        out_df.at[i, 'pred_label'] = 1
    elif pred_label == 'neutral' or pred_label == 'Neutral':
        out_df.at[i, 'pred_label'] = 0
    elif pred_label == 'negative' or pred_label == 'Negative':
        out_df.at[i, 'pred_label'] = -1
    else:
        print("problem with review: " , review ," ->", pred_label)

out_df[['sample_id']] = out_df[['sample_id']].astype(int)
out_df[['sample_id']] = out_df[['sample_id']].astype(int)
out_df.to_csv('prompt_out_tweet_preprocessed.csv', index=False)