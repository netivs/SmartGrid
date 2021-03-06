from pandas import read_csv
import numpy as np
from constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    weekend_holiday = read_csv('../data/weekend_holiday.csv')
    data_load_area = read_csv('../data/data_load_area.csv', usecols=LOAD_AREAS)
    np.savez('../data/data_load_area.npz', data = data_load_area, weekend_holiday = weekend_holiday['weekend_holiday'])
    test_data = read_csv('../data/test_data.csv')
    np.savez('../data/test_data.npz', data = test_data['NYISO'])

    day_2014 = read_csv('../data/day_2014.csv')
    np.savez('../data/day_2014.npz', data = day_2014['mw'])

    day_2015 = read_csv('../data/day_2015.csv')
    np.savez('../data/day_2015.npz', data = day_2015['mw'])