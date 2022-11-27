import csv
import pandas as pd

def get_human_data(file_name):
    data = []
    dataframe = pd.read_csv(file_name)
    return dataframe 

def stamp_human_data(filename):
    dataframe = pd.read_csv(filename)
    dataframe.insert(0,'sample_id',dataframe.index)
    return dataframe 

def filter_true_false(filename):
    df = pd.read_csv(filename)
    df['pred_label'] = df['pred_label'] + 1
    df.to_csv('all.csv', index=False)

    trues = df[df['pred_label'] == df['three_way_labels']]
    trues.to_csv('trues.csv', index=False)
    falses = df[df['pred_label'] != df['three_way_labels']]
    falses.to_csv('falses.csv', index=False)

def get_user_data(filename, userid):
    out_df =  pd.DataFrame({})
    df = pd.read_csv(filename)
    for i,row in df.iterrows():
        for keyind in range(len((row[11:1500] == row.three_way_labels).keys())):
            key = (row[11:1500] == row.three_way_labels).keys()[keyind]
            if key == userid:
                out_df.at[i, 'sample_id'] = row['sample_id']
                out_df.at[i, 'content']   = row['content']
                out_df.at[i, 'gold_label'] = row['three_way_labels']
                out_df.at[i, 'best_user_label'] = row[userid]
                avg_human_acc = row['average_accuracy']
                out_df.at[i, 'avg_human_acc'] = avg_human_acc
                if (row[11:1500] == row.three_way_labels)[keyind]:
                    #out_df.at[i, key+'_acc'] = True
                    out_df.at[i, 'best_human_acc'] = True
                else:
                    #out_df.at[i, key+'_acc'] = False
                    out_df.at[i, 'best_human_acc'] = False
    

if __name__=="__main__":
    filename = 'data/human/linguistic-acceptability/human_judgments.csv'
    dataframe = stamp_human_data(filename)
    dataframe.to_csv(filename, index=False)
    #dataframe = filter_true_false(filename)
 


