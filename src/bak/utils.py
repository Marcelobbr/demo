import os
import pandas as pd
import re

def remove_special_char(text):
    text = re.sub('[áàãâ]', 'a', text)
    text = re.sub('[óòõô]', 'o', text)
    text = re.sub('[éèê]', 'e', text)
    text = re.sub('[íì]', 'i', text)
    text = re.sub('[úù]', 'u', text)
    text = re.sub('ç', 'c', text)
    return text

def capture_data(inputs, file, label=False):
    path = os.path.join(inputs, file)
    return pd.read_csv(path, index_col='id')

def build_data_dict(inputs, dfs):
    dfs_dict = {}
    print('loading data into dictionary')
    for data_section in dfs:
        file = data_section+'.csv'
        dfs_dict[data_section] = capture_data(inputs, file)
    return dfs_dict

# def reorder_label_cols(df, targets):
#     cols = df.columns.to_list()

#     for target in targets:
#         cols.remove(target)
#         cols.append(target)
    
#     print('last columns:', df.columns[-3:])
#     return df[cols]

def reorder_label_cols(df, target):
    cols = df.columns.to_list()
    cols.remove(target)
    cols.append(target)
    
    print('last columns:', cols[-3:])
    return df[cols]


### preprocessing steps ###
def print_missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent of missing'])
    print(missing_data)