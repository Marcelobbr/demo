#!/usr/bin/env python
# coding: utf-8

# # Encoding

import numpy as np
import os
import pandas as pd
import re
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder


def ordinal_encode(df, testing = False):
    df = df.copy()
    categories_dict = {}

    for cat in df.columns:
        if df[cat].dtypes == 'object':
            categories_dict[cat] = list(df[cat].unique())
            if testing:
                print("Numero de categorias para variavel '{}': {} ".format(cat,df[cat].unique().size))

    if testing:
        print(list(categories_dict.keys()))

    enc = OrdinalEncoder(categories=list(categories_dict.values()))
    df[list(categories_dict.keys())] = enc.fit_transform(
        df[list(categories_dict.keys())])

    if testing:
        print(categories_dict)

    return df

def one_hot_encode(df, testing = False):
    df = df.copy()
    df_encoded = pd.get_dummies(df, prefix_sep='_', drop_first=True)
    
    # rename columns to show which are dummies
    onehot_cols = list(set(df_encoded.columns.to_list()) - set(df_encoded.columns.to_list()))
    onehot_cols_renaming = {col: 'dummy_'+col.replace('-', '_') for col in onehot_cols}
    df_encoded.rename(columns = onehot_cols_renaming, inplace=True)
    
    if testing:
        print('Quantity of columns before one-hot encoding:', len(df.columns))
        print('Quantity of columns after one-hot encoding:', len(df_encoded.columns))
        print('\r\nColumns of the new database:')
        print(df_encoded.columns.to_list())

    return df

# def one_hot_encode(df, testing = False):
#     df = df.copy()
#     df_encoded = pd.get_dummies(df, prefix_sep='_', drop_first=True)
    
#     if testing:
#         print('Quantity of columns before one-hot encoding:', len(df.columns))
#         print('Quantity of columns after one-hot encoding:', len(df_encoded.columns))
#         print('\r\nColumns of the new database:')
#         print(df_encoded.columns.to_list())
#     return df_encoded