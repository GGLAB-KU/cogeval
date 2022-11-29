import json
import pandas as pd

def log_question_details():
    fileObject = open("data/machine/reading-comprehension/multiRC/dev_83-fixedIds.json", "r")
    jsonContent = fileObject.read()
    obj = json.loads(jsonContent)
    df = pd.DataFrame({'sample_id': []})
    sid = 0
    insts = []
    for data in obj['data']:
        paragraph = data['paragraph']['text']
        questions = data['paragraph']['questions']
        for quest in questions:
            question = quest['question']
            answers = quest['answers']
            for ans in answers:
                answer = ans['text']
                label = ans['isAnswer']
                inst= (paragraph, question, answer, label)
                insts.append(inst)
                df.at[sid, 'sample_id'] = sid
                df.at[sid, 'paragraph'] = paragraph
                df.at[sid, 'question'] = question
                df.at[sid, 'answer'] = answer
                df.at[sid, 'label'] = int(label)
                sid+=1
        df[['sample_id']] = df[['sample_id']].astype(int)
    df.to_csv('multirc.csv', index=False)

def merge_files():
    meta_df = pd.read_csv('multirc.csv')
    machine_df = pd.read_csv('data/machine/reading-comprehension/multiRC/RoBERTa/multirc_roberta_calib.csv')

    for idx, row in machine_df.iterrows():
        paragraph = meta_df[meta_df.sample_id==row['sample_id']].paragraph
        question = meta_df[meta_df.sample_id==row['sample_id']].question
        answer = meta_df[meta_df.sample_id==row['sample_id']].answer

        machine_df.at[idx,'paragraph'] = paragraph.values[0]
        machine_df.at[idx,'question'] = question.values[0]
        machine_df.at[idx,'answer'] = answer.values[0]
    machine_df.to_csv('with_meta.csv', index=False)


if __name__ == '__main__':
    merge_files()






