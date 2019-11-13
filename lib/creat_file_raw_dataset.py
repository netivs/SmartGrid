from pandas import read_csv
import numpy as np
from constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    weekend_holiday = read_csv('../data/weekend_holiday.csv')
    np.savez('../data/weekend_holiday.csv', weekend_holiday = weekend_holiday['weekend_holiday'])
    data_load_area = read_csv('../data/data_load_area.csv', usecols=LOAD_AREAS)
    np.savez('../data/data_load_area.npz', data = data_load_area, weekend_holiday = weekend_holiday['weekend_holiday'])
