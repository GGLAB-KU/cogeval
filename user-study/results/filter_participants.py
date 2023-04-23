import pandas as pd

nonfiltered_df = pd.read_csv('user-study/results/nonfiltered_results_March7.csv')
filtered_df = pd.read_csv('user-study/results/filtered_results_Feb20.csv')
new_df = pd.DataFrame({})
for idx, row in nonfiltered_df.iterrows():
    if idx == 1:
        continue
    if idx == 0:
        new_df = new_df.append(row)
    if row.ResponseId in filtered_df[filtered_df.columns[5]].values:
        new_df = new_df.append(row)
new_df.to_csv('user-study/results/filtered_results_March7.csv')
