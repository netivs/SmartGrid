import numpy as np
from pmdarima.arima import auto_arima

from utils.constant import LOAD_AREAS
from utils.model import cal_error, binary_matrix, cal_error_all_load_areas


# l: The number of time-steps used for each prediction
# h: The number of time-steps need to be predicted
# p: The percentage size of train data compared to the whold data


def model_arima(data, l=24, r=0.8, h=3, p=0.6):
    # this list contains tuple {'name_of_load_area', 'gt list', 'prediction list'} to get all metrics of all load_ares
    metrics_all_load_area = []

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

        for t in range(len(test) - h + 1):
            # Only use l time-steps as inputs
            model = auto_arima(np.array(history[-l:]), error_action = 'ignore')
            yhat = model.predict(n_periods = h)
            predictions.append(yhat)
            gt.append(test[t:t+h])
            for i in range(h):
                if bm[(t+ h + i), LOAD_AREAS.index(load_area)] == 1:
                    # Update the data if verified == True
                    history.append(test[t+i])
                else:
                    # Otherwise use the predicted data
                    history.append(yhat[i])

        predictions = np.stack(predictions, axis=0)
        gt = np.stack(gt, axis=0)
        metric_load_area = (load_area, gt.flatten(), predictions.flatten())
        metrics_all_load_area.append(metric_load_area)

    cal_error_all_load_areas(metrics_all_load_area)