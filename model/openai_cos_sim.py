
from openai.embeddings_utils import cosine_similarity, get_embedding
import openai
import pandas as pd
import numpy as np
from scipy.stats import entropy

openai.api_key = "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" 
engine = 'text-similarity-babbage-001'

dframe = pd.read_csv("data/human/sentiment-analysis/lalor/sa_lalor_human.csv")
out_df =  pd.DataFrame({'sample_id':[],  'content':[], 'gold_label':[], 'pred_label':[], 'pred_confidence':[],  'gold_confidence':[], 'uncertainty':[]})
labels = ['Negative', 'Neutral', 'Positive']
label_embeddings = [get_embedding(label, engine=engine) for label in labels]

correct = 0
for (index, row) in dframe.iterrows():
    breakpoint()
    if row['sample_id']<10:
        continue
    if row['sample_id']>20:
        break
    input_embedding = openai.Embedding.create(
    input=row["content"],
    engine=engine
    )["data"][0]["embedding"]

    neg_score   = cosine_similarity(input_embedding, label_embeddings[0])
    neutr_score = cosine_similarity(input_embedding, label_embeddings[1])
    pos_score   = cosine_similarity(input_embedding, label_embeddings[2])
    all_scores = neg_score + neutr_score + pos_score
    neg_prob = neg_score/all_scores
    neutr_prob = neutr_score/all_scores
    pos_prob = pos_score/all_scores
    probs = [neg_prob, neutr_prob, pos_prob]
    
    pred_label =  np.argmax(probs) - 1
    pred_confidence = max(probs)
    
    gold_label = row['three_way_labels']
    if pred_label == gold_label:
        correct+=1
    gold_confidence = probs[gold_label+1]
    uncertainty = entropy(probs, base=2)
    out_df.loc[len(out_df.index)] = [row['sample_id'], row['content'], gold_label, pred_label, pred_confidence, gold_confidence, uncertainty]


out_df.to_csv('out_cos1.csv', index=False)
#dframe = pd.read_csv("out_cos.csv")
correct = 0
for (index, row) in out_df.iterrows():
    if row['pred_label'] == row['gold_label']:
        correct +=1
print(correct)
