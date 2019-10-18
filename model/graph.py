import numpy as np
from utils.constant import LOAD_AREAS, TRANSMISSION_ZONES

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

