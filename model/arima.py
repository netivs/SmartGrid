import numpy as np
import os
import time
import warnings
from lib import utils
from pmdarima.arima import auto_arima

class Arima():

    def __init__(self, **kwargs):

        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._alg_name = kwargs.get('alg')
        # data args
        self._raw_dataset_dir = self._data_kwargs.get('raw_dataset_dir')
        self._test_size = self._data_kwargs.get('test_size')

        # logging
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._logger.info(kwargs)

        # Model's Args
        self._verified_percentage = self._model_kwargs.get('verified_percentage')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._horizon = self._model_kwargs.get('horizon')
        self._nodes = self._model_kwargs.get('num_nodes')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')

        # Load data
        self._data = np.load(self._raw_dataset_dir)['data']

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['test'].get('log_dir')
        if log_dir is None:
            p = kwargs['data'].get('test_size')
            l = kwargs['model'].get('seq_len')
            h = kwargs['model'].get('horizon')
            r = kwargs['model'].get('verified_percentage')

            run_id = '%.1f_%d_%d_%.1f/' % (p, l, h, r)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _predict(self):
        bm = utils.binary_matrix(self._verified_percentage, self._data.shape[0], self._nodes)
        predictions = list()
        gt = []
        l = self._seq_len
        h = self._horizon
        
        # run predict for 29 nodes
        for column_load_area in range(self._nodes):
            data = self._data[:, column_load_area]
            size = int(len(data) * self._test_size)
            train, test = data[0:size], data[size:]
            history = [x for x in train]
            history = history[-l:]
            for t in range(0, len(test)-h, h):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Only use l time-steps as inputs                    
                    model = auto_arima(np.array(history[-l:]), error_action = 'ignore')
                    yhat = model.predict(n_periods = h)
                    predictions.append(yhat)
                    gt.append(test[t:t+h])
                    for i in range(h):
                        if bm[(t + size + i), column_load_area] == 1:
                            # Update the data if verified == True
                            history.append(test[t+i])
                        else:
                            # Otherwise use the predicted data
                            history.append(yhat[i])
        predictions = np.stack(predictions, axis=0)
        gt = np.stack(gt, axis=0)

        # save metrics to log
        error_list = utils.cal_error(gt.flatten(), predictions.flatten())
        utils.save_metrics(error_list, self._log_dir, self._alg_name)

    def test(self):
        for _ in range(self._run_times):
            self._predict()