import yaml, argparse, os
import seaborn as sns
import matplotlib.pyplot as plt
from yaml.loader import SafeLoader
from data.read_human import *
from data.read_machine import *
plt.switch_backend('Agg')

## works with pandas dataframe object
def correlate(dframe, cols, method):
    col_df = dframe[cols]
    corr = col_df.corr(method = method)
    return corr 

def save_heatmap(corr, name, title):
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='RdBu_r')
    heatmap.set_title(title)
    plt.xticks(rotation = 60)
    plt.savefig(name, dpi=600, bbox_inches='tight')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    args = parser.parse_args()
    
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=SafeLoader)
    
    # columns to correlate
    cols_human   = config['correlation']['columns']['human']
    cols_machine = config['correlation']['columns']['machine']
    method_corr  = config['correlation']['method']
    type_corr    = config['correlation']['type']
    file_human   = config['data']['human']
    file_machine = config['data']['machine']
    task = config['data']['type']
    
    if type_corr == 'human':
        cols_corr =  cols_human
        data_corr   = get_human_data(file_human)
    elif type_corr == 'machine':
        cols_corr =  cols_machine  
        data_corr = read_data(file_machine)
    elif type_corr == 'human+machine':
        cols_corr =  cols_human + cols_machine
        data_human   = get_human_data(file_human)
        data_machine = read_data(file_machine)
        # TODO: fix this workaround, to take machine-only data
        #data_machine = data_machine[['sample_id','pred_label', 'gold_label', 'pred_confidence', 'gold_confidence', 'uncertainty']]
        #data_machine = data_machine[['sample_id','pred_label', 'confidence']]
        
        data_corr    = pd.concat([data_human, data_machine], axis=1)
        data_corr = pd.merge(data_human, data_machine, on="sample_id")
    else:
        print('correlation type should be (1) human+machine, (2) human or (3) machine.')
        return
  
    #drop duplicate columns
    #data_corr_2 = data_corr.loc[:,~data_corr.T.duplicated(keep='first')]
    # correlate with specified method
    corr         = correlate(data_corr, cols_corr, method_corr)
    num_samples = len(data_corr)
    print("# of samples for correlation: %d" % num_samples)
    # save correlation results
    results_dir  = config['correlation']['results']['dir']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    excel_name = results_dir+config['correlation']['results']['excel']['name']
    corr.to_excel(excel_name)
    heatmap_name  = results_dir + config['correlation']['results']['heatmap']['name']
    heatmap_title = config['correlation']['results']['heatmap']['title'] + " #samples: " + str(num_samples)
    heatmap      = save_heatmap(corr, heatmap_name, heatmap_title)


if __name__=="__main__":
    # specify parameters in .yaml file under ./config directory 
    # usage: python correlation.py <config_file>
    # e.g. : python correlation.py config/sa_all.yaml
    main()
    