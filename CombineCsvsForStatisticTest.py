import pandas as pd
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

file_name_to_csvs = {}
folders = glob.glob('*')
for folder in folders:
    if "Data" in folder:
        files = glob.glob(folder + '/*.csv')
        for file in files:
            file_name = file.split('\\')[-1]
            if file_name in file_name_to_csvs:
                file_name_to_csvs[file_name] = pd.concat([file_name_to_csvs[file_name], pd.read_csv(file, index_col=0)],
                                                         axis=0)
            else:
                file_name_to_csvs[file_name] = pd.read_csv(file, index_col=0)

for k, v in file_name_to_csvs.items():
    v.reset_index(drop=True)
    print(k)
    v.to_csv('Combined/' + k)

finalCsv = None

for k in range(1, 8):
    file1 = 'Combined/Lookahead k={}.csv'.format(k)
    file2 = 'Combined/LookaheadNew k={}.csv'.format(k)

    csv1 = pd.read_csv(file1)
    csv1.rename(columns={'Time': 'Time AL*', 'Expanded DFS': 'Expanded DFS AL*', 'Expanded A*': 'Expanded A* AL*',
                         'Length': 'Length AL*'}, inplace=True)

    csv2 = pd.read_csv(file2)
    csv2.rename(columns={'Time': 'Time ALC*', 'Expanded DFS': 'Expanded DFS ALC*', 'Expanded A*': 'Expanded A* ALC*',
                         'Length': 'Length ALC*'}, inplace=True)

    together = pd.concat([csv1, csv2], axis=1)

    #
    for index, row in together.iterrows():
        if row['Length AL*'] != row['Length ALC*']:
            together = together.head(index)
            break

    #

    if finalCsv is None:
        finalCsv = together
    else:
        finalCsv = pd.concat([finalCsv, together], axis=0)

finalCsv.to_csv('StatisticalCsv/result-AL*vsALC*.csv')
