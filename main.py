from pandas import read_csv
<<<<<<< HEAD
from model.arima import model_arima
=======
>>>>>>> 5187077e34a1861fdd61bb5b54dd6fad764b5e2a
from model.dcrnn import prepare_data
from utils.constant import LOAD_AREAS

if __name__ == "__main__":
    # read file skip datetime column
    data_load_area = read_csv('data/data_load_area.csv', usecols=LOAD_AREAS)
    x, y = prepare_data(data_load_area, p=0.3)
    print(x.shape)

