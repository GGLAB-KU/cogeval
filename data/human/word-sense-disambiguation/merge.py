import pandas as pd

df = pd.read_csv('data/human/word-sense-disambiguation/ann3_4_wic.csv')

breakpoint()

for idx, row in df.iterrows():
    if row['gold_label'] == 'T':
        row['gold_label'] = 1
    else:
        row['gold_label'] = 1
