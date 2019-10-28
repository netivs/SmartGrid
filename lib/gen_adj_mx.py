from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
from pandas import read_csv

from constant import LOAD_AREAS, TRANSMISSION_ZONES

def construct_graph(data_grid, nodes = 29):
    undirected_graph = np.zeros((nodes,nodes), dtype = np.float32)

    # get companies by zone
    for zone in TRANSMISSION_ZONES:
        data_companies = data_grid[data_grid['zone'] == zone]
        unique_companies = data_companies['load_area'].unique()
        len_companies = len(unique_companies)

        # connect node
        for i in range(0, len_companies-1):
            for j in range(i+1, len_companies):
                if i == j:
                    continue
                row = LOAD_AREAS.index(unique_companies[i])
                col = LOAD_AREAS.index(unique_companies[j])
                undirected_graph[row][col] = 1
                undirected_graph[col][row] = 1
    print(undirected_graph)
    return undirected_graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='data/data_grid.csv',
                        help='Read input from csv file.')
    parser.add_argument('--output_pkl_filename', type=str, default='data/dcrnn/adj_mx.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    data_load_area = read_csv(args.csv_file)
    adj_mx = construct_graph(data_load_area)
    # Save to pickle file.
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump(adj_mx, f, protocol=2)
