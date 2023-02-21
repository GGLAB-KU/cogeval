import openai, time
import pandas as pd
import numpy as np
openai.api_key = "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" 

df = pd.read_csv("data/machine/reading-comprehension/multiRC/meta.csv")
out_df = pd.DataFrame({'sample_id': [], 'paragraph': [], 'question': [], 'answer-option': [], 'pred_label': [], 'gold_label': [], 'logprobs':[]})

for i,row in df.iterrows():
    print(row['sample_id'])
    if 200 > row['sample_id']:
        continue
    if row['sample_id'] > 300:
        break
    
    out_df.at[i, 'sample_id'] = row['sample_id']
    paragraph = row['paragraph']
    question = row['question']
    answer_option = row['answer']
    out_df.at[i, 'paragraph'] = row['paragraph']
    out_df.at[i, 'question'] = row['question']
    out_df.at[i, 'answer-option'] = row['answer']
    out_df.at[i, 'gold_label'] = row['label']

    prompt_text = paragraph+ "\n\n\n" + "Question:" + question + "\n\n" + "Can we say " + answer_option + "?" + "Answer as yes or no.\n\n" + "Answer:"
    
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= prompt_text,
    temperature=0.0,
    max_tokens=1,
    top_p=1,
    frequency_penalty=0.0,
    logprobs = 3,
    presence_penalty=0)
    confidence =  np.exp(response['choices'][0]['logprobs']['token_logprobs'][0])
    out_df.at[i, 'confidence'] = confidence
    out_df.at[i, 'logprobs'] = str(response['choices'][0]['logprobs']['top_logprobs'][0])

    pred_label =  response.choices[0].text.strip()
    if pred_label == 'Yes' or pred_label == 'yes' or pred_label == 'yes.' or pred_label == 'Yes.':
        out_df.at[i, 'pred_label'] = 1
    elif pred_label == 'No' or pred_label == 'no' or pred_label == 'no.' or pred_label == 'No.':
        out_df.at[i, 'pred_label'] = 0
    else:
        print("problem with question: " , question ," option: ", answer_option, "->", pred_label)

    if row['sample_id'] %25 == 0 and row['sample_id']!=0:
        out_df[['sample_id']]  = out_df[['sample_id']].astype(int)
        out_df[['gold_label']] = out_df[['gold_label']].astype(int)
        out_df.to_csv('multirc_davinci-zeroshot_'+str(row['sample_id'])+'.csv', index=False)
        time.sleep(60)