import pmdarima as pm
from utils.constant import LOAD_AREAS
from utils.model import cal_error, binary_matrix
import matplotlib.pyplot as plt

# l: The number of time-steps used for each prediction
# h: The number of time-steps need to be predicted
# p: The percentage size of train data compared to the whold data
def model_arima(data, nodes, l, r = 0.8, h = 1, p = 0.6):
    binary_matrix = binary_matrix(r, nodes, l)
    # run predict for 29 nodes
    for load_area in LOAD_AREAS:
        series = data[[load_area]].squeeze('columns')
        X = series.values
        size = int(len(X) * p)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(train) - l -h):
            model = pm.auto_arima(history, error_action = 'ignore', seasonal = True, m = l)
            output = model.predict(n_periods = h)
            
            ####### STUCK HERE. dont know what to do next when having H of outputs
            yhat = output[0]
            predictions.append(yhat)
            if binary_matrix[LOAD_AREAS.index(load_area)][t+size] == 1:
                obs = test[t]
            else:
                history.append(test[t])
            ##########

            print('predicted=%f, expected=%f' % (yhat, obs))
        
        cal_error(test, predictions)

        # plot
        plt.plot(test)
        plt.plot(predictions, color='red')
        plt.show()