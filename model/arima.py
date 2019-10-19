import numpy as np
from pmdarima.arima import auto_arima

from utils.constant import LOAD_AREAS
from utils.model import cal_error, binary_matrix


# l: The number of time-steps used for each prediction
# h: The number of time-steps need to be predicted
# p: The percentage size of train data compared to the whold data


def model_arima(data, l=24, r=0.8, h=3, p=0.6):

    bm = binary_matrix(r, len(LOAD_AREAS), data.shape[0])

    # run predict for 29 nodes
    for load_area in LOAD_AREAS:
        series = data[[load_area]].squeeze('columns')
        if series.empty == True:
            continue
        X = series.values
        size = int(len(X) * p)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]

        # Cant pass history[-l:] directly to auto_arima
        histotry = history[-l:]
        predictions = list()
        gt = []

        for t in range(len(train) - l - h):

            # Only use l time-steps as inputs
            model = auto_arima(np.array(history[-l:]), error_action='ignore')
            yhat = model.predict(n_periods = h)
            predictions.append(yhat)
            gt.append(test[t:t+h])
            for i in range(h):
                if bm[(t+size + i), LOAD_AREAS.index(load_area)] == 1:
                    # Update the data if verified == True
                    history.append(test[t+i])
                else:
                    # Otherwise use the predicted data
                    history.append(yhat[i])

        predictions = np.stack(predictions, axis=0)
        gt = np.stack(gt, axis=0)
        cal_error(gt.flatten(), predictions.flatten())

        # plot
        # plt.plot(test)
        # plt.plot(predictions, color='red')
        # plt.show()