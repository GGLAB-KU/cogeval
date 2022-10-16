import csv
from collections import Counter
import pandas as pd

def read_sentiment(fname):
    '''data = []
    with open(fname) as csvfile:
        rdr = csv.reader(csvfile, delimiter=',')
        for row in rdr:
            if row[0]=='sstb_id':
                keys = row
            else:
                instance = {}
                for key,value in zip(keys,row):
                    instance[key] = value
                data.append(instance)
    return data'''
    # easier to use dataframe
    dataframe = pd.read_csv(fname)
    return dataframe



def get_splitted_data(task="sentiment"):
    """
    Get the split info for ZuCo - SSTB
    """
    # split meanings: 1: train 2: test 3: dev
    train_data = []
    val_data = []
    test_data = []

    if task == "sentiment":
        # create train, dev and test splits with this structure
        human_data = read_human_csv.get_data("human/sentiment/sentiment_human_control.csv", task="sentiment")

        with open("machine/sst/datasetSplit.txt") as f:
            splits = [line.split(',')[1] for line in f.read().splitlines()]
        f.close()

        # splits[index] corresponds to the split number of sentence with id index
        split_lst = []
        for hum_meas in human_data:
            split_ind = splits[int(hum_meas["sstb_id"])]
            split_lst.append(split_ind)
            if split_ind == '1':
                train_data.append(hum_meas)
            elif split_ind == '2':
                test_data.append(hum_meas)
            elif split_ind == '3':
                val_data.append(hum_meas)

        # 32 sentences from training, 12 from test, 2 from validation
        # print(Counter(split_lst))

    elif task == "re":
        print("all from validation data. no need to align")
        # 400 from validation sentences. but there is no model anyway
    elif task == "qa":
        print("all from test data. no need to align")
    else:
        print("no need to align")

    print("debug")
    return train_data, val_data, test_data


def make_new_splits(train, dev, test):
    # remove the human train, dev from the original train, dev and put them into test
    # return the train, dev, test files

    sents = []

    sent_info = dict()

    # 1: read the original sentences (starts from 1)
    with open("machine/sst/datasetSentences.txt") as f:
        sents = [line.split('\t')[1] for line in f.read().splitlines()]
    f.close()

    # 2: get the splits each sentence belongs
    with open("machine/sst/datasetSplit.txt") as f:
        splits = [line.split(',')[1] for line in f.read().splitlines()]
    f.close()

    # 3: get the labels for each sentence in the original splits
    with open("machine/sst/train/dlabels.txt") as f:
        train_labs = [line[-1] for line in f.read().splitlines()]
    f.close()
    with open("machine/sst/dev/dlabels.txt") as f:
        dev_labs = [line[-1] for line in f.read().splitlines()]
    f.close()
    with open("machine/sst/test/dlabels.txt") as f:
        test_labs = [line[-1] for line in f.read().splitlines()]
    f.close()
    print("debug")


def get_machine_data(config):
    file_name = config['data']['machine']
    task = config['data']['type']
    data = []
    if task == "sentiment":
        data = read_sentiment(file_name)
    #elif task == "re":
    #    data = read_re(file_name)
    return data



#if __name__ == '__main__':
    #train, dev, test = get_splitted_data(task="sentiment")
    #make_new_splits(train, dev, test)

