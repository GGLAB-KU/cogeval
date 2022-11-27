import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from openai.embeddings_utils import get_embedding
import openai, time

openai.api_key = "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" 


## Get training data
datafile_path = "data/2900_embeddings_PART1.csv"  
df = pd.read_csv(datafile_path)
df["embs"] = df.embs.apply(eval).apply(np.array)

df.label = df.label+1
X_train, X_test, y_train, y_test = train_test_split(list(df.embs.values), df.label, test_size=0.1, random_state=42)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

## Get test results
input_datapath = 'data/human/sentiment-analysis/lalor/sa_lalor_human.csv' 
true  = 0; count = 0
engine = "text-similarity-babbage-001"
out_df = pd.DataFrame({'sample_id': [], 'pred_label': [], 'gold_confidence': []})
human = pd.read_csv(input_datapath)
for (index, row) in human.iterrows():
    print(index)
    if index % 20 ==0:
        time.sleep(60)
    test_embedding = openai.Embedding.create(input=row["content"], engine=engine)["data"][0]["embedding"]
    pred = clf.predict([test_embedding])
    probas = clf.predict_proba([test_embedding])
    gold_confidence = probas[0][row['three_way_labels']+1]
    pred_confidence = max(probas[0])

    pred = pred[0] -1
    if pred == row['three_way_labels']:
        true+=1
    print("sample_id: " + str(row['sample_id']) , " pred_label: " + str(pred) , " confidence: " , gold_confidence)
    count+=1
    #print("content: " , row['content'] , " pred: " , pred, " three_way_pred: ", three_way_pred)
    out_df.at[index,'sample_id'] = row['sample_id']
    out_df.at[index,'pred_label'] = pred
    out_df.at[index,'gold_confidence'] = gold_confidence
    out_df.at[index,'pred_confidence'] = pred_confidence
    out_df[['sample_id']] = out_df[['sample_id']].astype(int)
    out_df[['pred_label']] = out_df[['pred_label']].astype(int)

breakpoint()
print('# trues: ', true)
print('# total: ', count)
out_df.to_csv('classifier.csv')

'''offset = 600
for j in range(10):
    offset += 50
    print(offset)
    _human = human[offset-50:offset]
    _human['babbage_similarity'] = _human.content.apply(lambda x: get_embedding(x, engine='text-similarity-babbage-001'))
    _human.to_csv('data/human'+str(offset)+'.csv')
    time.sleep(60)'''

#report = classification_report(y_test, preds)
#print(report)