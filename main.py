from pandas import read_csv
from model.dcrnn import prepare_data
from utils.constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_load_area.csv', usecols=LOAD_AREAS)
    x, y = prepare_data(data_load_area, p=0.3)

