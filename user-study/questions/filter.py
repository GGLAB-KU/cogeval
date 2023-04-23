import pandas as pd

fdf = pd.DataFrame({})
df = pd.read_csv('/kuacc/users/mugekural/workfolder/dev/git/cogeval/results/SNLI-lalor/v3_all.csv')
for idx,row in df.iterrows():
    skip = False
    threshold = 0.95
    if (row['machine_2_pred'] == row['machine_3_pred']) :
        if row['machine_2_confidence'] > threshold and row['machine_3_confidence'] > threshold:
            skip = True
    if (row['machine_2_pred'] == row['machine_4_pred']) :
        if row['machine_2_confidence'] > threshold and row['machine_4_confidence'] > threshold:
            skip = True   
    if (row['machine_3_pred'] == row['machine_4_pred']) :
        if row['machine_3_confidence'] > threshold and row['machine_4_confidence'] > threshold:
            skip = True   
    if not skip:
        fdf = fdf.append(row)

fdf.to_csv('results/SNLI-lalor/v3_fdf95_Y.csv', index=False)         